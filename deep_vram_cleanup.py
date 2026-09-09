"""Deep VRAM Cleanup - Per-source toggle cleanup for ComfyUI."""

import ctypes
import gc
import logging
import os
import sys
from typing import Tuple, List, Dict, Any

import torch
import comfy.model_management as mm
from comfy.model_management import (
    soft_empty_cache,
    unload_all_models,
    cleanup_models,
    cleanup_models_gc,
)


# ---------------------------------------------------------------------------
#  Third-party caches  (leak sources to clear)
# ---------------------------------------------------------------------------

_RIFE_CACHE = None          # RIFE _model_cache
_BACKWARP_GRID = None        # RIFE backwarp_tenGrid
try:
    from vfi_models.rife import _model_cache as _rife_cache
    _RIFE_CACHE = _rife_cache
except Exception:
    pass
try:
    from vfi_models.rife.rife_arch import backwarp_tenGrid as _grid
    _BACKWARP_GRID = _grid
except Exception:
    pass


# ---------------------------------------------------------------------------
#  cudaMallocAsync pool trimmer
# ---------------------------------------------------------------------------

def _find_cudart_dll():
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        for name in ("cudart64_13.dll", "cudart64_130.dll", "cudart64_12.dll"):
            p = os.path.join(torch_lib, name)
            if os.path.exists(p):
                return ctypes.CDLL(p)
    except Exception:
        pass
    for name in ("cudart64_13.dll", "cudart64_130.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _trim_cuda_mempool(device_idx: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cudart = _find_cudart_dll()
    if cudart is None:
        return False
    try:
        cudaDeviceGetMemPool = cudart.cudaDeviceGetMemPool
        cudaDeviceGetMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cudaDeviceGetMemPool.restype = ctypes.c_int
        cudaMemPoolTrimTo = cudart.cudaMemPoolTrimTo
        cudaMemPoolTrimTo.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cudaMemPoolTrimTo.restype = ctypes.c_int
    except AttributeError:
        return False
    mempool = ctypes.c_void_p()
    if cudaDeviceGetMemPool(ctypes.byref(mempool), device_idx) != 0:
        return False
    cudaMemPoolTrimTo(mempool, 0)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return True


# ---------------------------------------------------------------------------
#  Cache clearing helpers
# ---------------------------------------------------------------------------

def _clear_all_caches_recursive(cache_obj, cleared_count: int = 0) -> int:
    total = cleared_count
    try:
        if hasattr(cache_obj, "cache") and isinstance(cache_obj.cache, dict):
            total += len(cache_obj.cache)
            cache_obj.cache.clear()
    except Exception:
        pass
    try:
        if hasattr(cache_obj, "subcaches") and isinstance(cache_obj.subcaches, dict):
            for _key, sub in list(cache_obj.subcaches.items()):
                total = _clear_all_caches_recursive(sub, total)
            cache_obj.subcaches.clear()
    except Exception:
        pass
    return total


def _find_and_clear_execution_caches() -> int:
    cleared = 0
    for obj in gc.get_objects():
        cls_name = type(obj).__name__
        if cls_name in ("BasicCache", "HierarchicalCache", "LRUCache", "RAMPressureCache"):
            n = _clear_all_caches_recursive(obj, 0)
            cleared += n
    return cleared


def _clear_vfi_model_caches(which: str = "all") -> Dict[str, int]:
    """Clear RIFE module-level caches.

    Moves tensors to CPU first to avoid PyTorch C++ warning storms.

    Args:
        which: "all", "grid" (backwarp_tenGrid only), or "model" (_model_cache only)
    """
    result: Dict[str, int] = {}

    if which in ("all", "model"):
        if _RIFE_CACHE is not None:
            try:
                n = 0
                for _key, model in list(_RIFE_CACHE.items()):
                    try:
                        model.cpu()
                    except Exception:
                        pass
                    n += 1
                _RIFE_CACHE.clear()
                result["RIFE _model_cache"] = n
            except Exception:
                pass

    if which in ("all", "grid"):
        if _BACKWARP_GRID is not None:
            try:
                n = 0
                for _key, t in list(_BACKWARP_GRID.items()):
                    try:
                        if torch.is_tensor(t) and t.is_cuda:
                            t.detach_()
                            t.data = torch.empty(0, device="cpu")
                    except Exception:
                        pass
                    n += 1
                _BACKWARP_GRID.clear()
                result["RIFE backwarp_tenGrid"] = n
            except Exception:
                pass

    for k, v in result.items():
        logging.info("[DeepVRAMCleanup] Cleared %s: %d entries", k, v)

    return result


def _has_safe_dict(obj) -> bool:
    try:
        d = object.__getattribute__(obj, "__dict__")
        return isinstance(d, dict) and bool(d)
    except Exception:
        return False


def _purge_cuda_orphans() -> int:
    removed = 0
    seen_ids: set[int] = set()
    for obj in gc.get_objects():
        oid = id(obj)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        if isinstance(obj, (type, type(print), type(sys))):
            continue
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                removed += 1
                obj.detach_()
        except Exception:
            pass
        if not _has_safe_dict(obj):
            continue
        try:
            for attr, val in list(obj.__dict__.items()):
                try:
                    if torch.is_tensor(val) and val.is_cuda:
                        obj.__dict__[attr] = val.detach().cpu()
                        removed += 1
                except Exception:
                    pass
        except Exception:
            pass
    return removed


# ---------------------------------------------------------------------------
#  VRAM -> system RAM offload (non-destructive)
# ---------------------------------------------------------------------------

def _offload_loaded_models_to_ram() -> List[str]:
    """Move loaded model weights from VRAM to system RAM, keeping the models loaded.

    Calls ComfyUI's own partial-offload path (ModelPatcher.partially_unload, the same
    one free_memory() uses under VRAM pressure): weights stay in (pinned) host RAM and
    are streamed back to VRAM automatically the next time the model runs.
    """
    gb = 1024 ** 3
    lines: List[str] = []
    for loaded in list(mm.current_loaded_models):
        patcher = loaded.model
        if patcher is None:
            continue
        resident = patcher.loaded_size()
        if resident <= 0:
            continue
        freed = patcher.partially_unload(patcher.offload_device, 1e32)
        name = patcher.model.__class__.__name__
        lines.append(f"offloaded {name}: {freed / gb:.3f} GB VRAM -> {patcher.offload_device} (was {resident / gb:.3f} GB)")
        logging.info("[DeepVRAMCleanup] Offloaded %s: %.1f MB VRAM -> %s", name, freed / (1024 ** 2), patcher.offload_device)
    return lines


# ---------------------------------------------------------------------------
#  Diagnostics
# ---------------------------------------------------------------------------

def _find_gpu_tensors_direct() -> List[str]:
    lines: List[str] = []
    tensors: List[torch.Tensor] = []
    objs = gc.get_objects()
    for obj in objs:
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append(obj)
        except Exception:
            pass
    del objs
    lines.append(f"total GPU tensors: {len(tensors)}")
    if not tensors:
        return lines
    for i, t in enumerate(tensors[:15]):
        shape = tuple(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
        size_mb = t.numel() * t.element_size() / (1024**2)
        lines.append(f"  [{i}] {shape} {dtype} {size_mb:.1f}MB")
    return lines


def _inspect_comfy_caches() -> List[str]:
    lines: List[str] = []
    gb = 1024**3
    try:
        loaded = mm.current_loaded_models
        lines.append(f"loaded_models: {len(loaded)}")
    except Exception:
        lines.append("loaded_models: (access failed)")

    try:
        import comfy_execution.caching as _caching
        for obj in gc.get_objects():
            if not isinstance(obj, _caching.BasicCache):
                continue
            n = len(getattr(obj, "cache", {}))
            s = len(getattr(obj, "subcaches", {}))
            gpu_n = 0
            gpu_b = 0
            for _k, v in list(getattr(obj, "cache", {}).items()):
                for t in ([v] if torch.is_tensor(v) else
                           v if isinstance(v, (list, tuple)) else []):
                    if torch.is_tensor(t) and t.is_cuda:
                        gpu_n += 1
                        gpu_b += t.numel() * t.element_size()
            lines.append(
                f"exec_cache {type(obj).__name__}: {n} entries {s} subs "
                f"{gpu_n} GPU ({gpu_b/gb:.3f} GB)"
            )
    except Exception:
        pass
    return lines


# ---------------------------------------------------------------------------
#  Node
# ---------------------------------------------------------------------------

class DeepVRAMCleanup:
    """Selective VRAM cleanup with per-source toggles.

    ======================  ==============================================
    execution_cache         Clear ComfyUI intermediate node output cache
    unload_models           Unload all loaded models (needs reload later)
    offload_models_to_ram   Move loaded model weights VRAM -> RAM (models stay loaded)
    rife_backwarp_grid      Clear RIFE backwarp_tenGrid (grid tensors)
    rife_model_cache        Clear RIFE _model_cache (cached IFNet)
    gc_empty_cache          gc.collect() + torch.cuda.empty_cache()
    trim_cuda_pool          Direct cudaMemPoolTrimTo(0) (cudaMallocAsync)
    purge_orphan_tensors    Hunt & detach stray CUDA tensors from heap
    ======================  ==============================================
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
                        "required": {
                "execution_cache": ("BOOLEAN", {"default": True,
                    "tooltip": "清空 ComfyUI 的中间节点输出缓存。每个节点跑完后的输出会被缓存以加速下次执行，但也因此持有大量中间 tensor 的引用。此开关清除所有已缓存的中间结果，不影响已加载的模型。"}),
                "unload_models": ("BOOLEAN", {"default": False,
                    "tooltip": "卸载 ComfyUI 模型管理中所有已加载的模型（checkpoint / LoRA / VAE 等）。清理后再次生成图像需要重新加载模型，会有短暂延迟。日常不建议开启，除非显存极度紧张。"}),
                "offload_models_to_ram": ("BOOLEAN", {"default": False,
                    "tooltip": "把已加载模型的权重从显存搬到系统内存，但不卸载模型。\n\n走的是 ComfyUI 自己的部分卸载路径（ModelPatcher.partially_unload，即显存不足时 free_memory 用的同一套机制）：权重留在（可锁页的）内存里，模型仍处于「已加载」状态，下次用到时自动回传到显存，不需要重新从磁盘加载。\n\n适合「采样跑完了，先腾出显存给放大/VAE，但不想丢掉 checkpoint」这类场景。代价是下次用到该模型时要走一次 PCIe 回传。注意：同时开启 unload_models 时，模型最终还是会被卸载。"}),
                "rife_backwarp_grid": ("BOOLEAN", {"default": True,
                    "tooltip": "清空 RIFE 插帧节点的 backwarp_tenGrid 模块级缓存。\n\nRIFE 的 warp() 函数会按 (分辨率, 批量大小) 缓存光流坐标网格（shape: batch×2×H×W 的 float32 CUDA tensor）。每处理一个不同分辨率的视频就会新增一个条目，且永远不会自动释放。\n\n这是 ComfyUI-Frame-Interpolation 最主要的显存泄漏源。清空后下次插帧会自动重建当前分辨率需要的网格。"}),
                "rife_model_cache": ("BOOLEAN", {"default": False,
                    "tooltip": "清空 RIFE 插帧节点的 _model_cache 模块级缓存。\n\nRIFE 会把加载的 IFNet 模型缓存在模块全局字典中，避免每次插帧重复加载。清空后下次插帧需要重新从磁盘加载模型（几秒延迟）。日常不建议开启。"}),
                "gc_empty_cache": ("BOOLEAN", {"default": True,
                    "tooltip": "执行 Python gc.collect() + torch.cuda.empty_cache()。释放 Python 层不再被引用的对象，并将 PyTorch CUDA 缓存池中已标记为空闲的内存归还给操作系统。"}),
                "trim_cuda_pool": ("BOOLEAN", {"default": True,
                    "tooltip": "直接调用 CUDA Runtime 的 cudaMemPoolTrimTo(0)，强制将 cudaMallocAsync 内存池中所有未使用的内存归还给操作系统。\n\n这是对 torch.cuda.empty_cache() 的补充——在 CUDA 13.x + cudaMallocAsync 后端下，仅靠 empty_cache() 往往无法彻底释放 reserved 内存。此选项直接绕过 PyTorch 操作 CUDA 内存池。"}),
                "purge_orphan_tensors": ("BOOLEAN", {"default": False,
                    "tooltip": "（暴力模式）遍历 Python 堆中所有 gc 追踪的对象，找出散落在类属性、全局变量等角落的 GPU tensor，强制 detach 或移动到 CPU。\n\n此操作慢且激进，可能影响其他节点的内部状态。仅在 deep 清理无效时作为最后手段使用。"}),
            },
            "optional": {
                "passthrough1": ("*", {"default": None, "forceInput": False}),
                "passthrough2": ("*", {"default": None, "forceInput": False}),
                "passthrough3": ("*", {"default": None, "forceInput": False}),
                "passthrough4": ("*", {"default": None, "forceInput": False}),
            },
        }

    RETURN_TYPES = ("*", "*", "*", "*", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passthrough1", "passthrough2", "passthrough3", "passthrough4",  "used_gb", "total_gb", "freed_gb", "diag")
    FUNCTION = "cleanup"
    CATEGORY = "utils/VRAM"

    def _vram(self, device: torch.device) -> Tuple[float, float]:
        if device.type != "cuda":
            return 0.0, 0.0
        gb = 1024**3
        a = torch.cuda.memory_allocated(device) / gb
        r = torch.cuda.memory_reserved(device) / gb
        return max(a, r), torch.cuda.get_device_properties(device).total_memory / gb

    def cleanup(
        self,
        execution_cache: bool,
        unload_models: bool,
        offload_models_to_ram: bool,
        rife_backwarp_grid: bool,
        rife_model_cache: bool,
        gc_empty_cache: bool,
        trim_cuda_pool: bool,
        purge_orphan_tensors: bool,
        passthrough1=None,
        passthrough2=None,
        passthrough3=None,
        passthrough4=None,
    ) -> Tuple[float, float, float, str]:
        device = mm.get_torch_device()
        used_before, total = self._vram(device)

        # --- Apply selected cleanups ---
        if gc_empty_cache:
            gc.collect()
            if device.type == "cuda":
                soft_empty_cache(force=True)

        if execution_cache:
            n = _find_and_clear_execution_caches()
            logging.info("[DeepVRAMCleanup] Cleared %d execution-cache entries", n)

        offload_lines: List[str] = []
        if offload_models_to_ram:
            offload_lines = _offload_loaded_models_to_ram()

        if unload_models:
            cleanup_models()
            cleanup_models_gc()
            unload_all_models()
            if device.type == "cuda":
                soft_empty_cache(force=True)

        # RIFE caches (selective)
        rife_which = (
            "all" if (rife_backwarp_grid and rife_model_cache)
            else "grid" if rife_backwarp_grid
            else "model" if rife_model_cache
            else ""
        )
        if rife_which:
            _clear_vfi_model_caches(which=rife_which)

        if purge_orphan_tensors:
            n = _purge_cuda_orphans()
            logging.info("[DeepVRAMCleanup] Purged %d orphan CUDA tensors", n)

        # Final GC + cache flush
        for _ in range(3):
            gc.collect()
        if device.type == "cuda":
            soft_empty_cache(force=True)

        if trim_cuda_pool and device.type == "cuda":
            _trim_cuda_mempool(device.index if hasattr(device, "index") else 0)

        # --- Diagnostics ---
        used_after, _ = self._vram(device)
        freed = max(0.0, used_before - used_after)

        lines = [
            f"used: {used_before:.3f} -> {used_after:.3f} GB  (freed {freed:.3f})",
            f"total: {total:.2f} GB",
            "",
        ]
        lines.extend(_inspect_comfy_caches())
        if offload_lines:
            lines.append("")
            lines.extend(offload_lines)
        lines.append("")
        lines.extend(_find_gpu_tensors_direct())

        diag = "\n".join(lines)
        return (passthrough1, passthrough2, passthrough3, passthrough4, used_after, total, freed, diag)


NODE_CLASS_MAPPINGS = {"DeepVRAMCleanup": DeepVRAMCleanup}
NODE_DISPLAY_NAME_MAPPINGS = {"DeepVRAMCleanup": "Deep VRAM Cleanup"}
