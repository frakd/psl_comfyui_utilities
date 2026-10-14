"""Deep VRAM Cleanup - Per-source toggle cleanup for ComfyUI."""

import ctypes
import gc
import logging
import os
import re
import sys
from typing import Tuple, List, Dict, Any

import torch
from comfy_api.latest import io
import comfy.model_management as mm
from comfy.model_management import (
    soft_empty_cache,
    unload_all_models,
    cleanup_models,
    cleanup_models_gc,
)

# 调试开关（纯后端）：环境变量 PSL_DEBUG=1 时输出运行轨迹日志，默认静默。
# warning/error 不受开关影响，始终可见。
log = logging.getLogger("psl_comfyui_utilities.deep_vram_cleanup")
if os.environ.get("PSL_DEBUG", "").lower() in ("1", "true", "yes"):
    log.setLevel(logging.DEBUG)

# 内核的选择性卸载接口（较新的 ComfyUI 才有）。缺失时退回本文件内置的 keep_loaded 实现。
try:
    from comfy.model_management import unload_model_and_clones as _unload_model_and_clones
except ImportError:
    _unload_model_and_clones = None
    log.debug(
        "[DeepVRAMCleanup] comfy.model_management.unload_model_and_clones 不可用，"
        "指定模型卸载将使用内置 keep_loaded 回退实现"
    )


# ---------------------------------------------------------------------------
#  Third-party caches  (leak sources to clear)
# ---------------------------------------------------------------------------
#
# ===== RIFE 相关支持已停用（用户不再用插帧节点）=====
# 下面两处 RIFE 缓存导入与 _clear_vfi_model_caches 已整体注释掉，define_schema 里对应
# 的 rife_backwarp_grid / rife_model_cache 两个开关也已删除。
# 如需恢复：取消本段与被注释函数的注释，并把 execute() 里的 rife_which 分支加回来。
#
# _RIFE_CACHE = None          # RIFE _model_cache
# _BACKWARP_GRID = None        # RIFE backwarp_tenGrid
# try:
#     from vfi_models.rife import _model_cache as _rife_cache
#     _RIFE_CACHE = _rife_cache
# except Exception:
#     pass
# try:
#     from vfi_models.rife.rife_arch import backwarp_tenGrid as _grid
#     _BACKWARP_GRID = _grid
# except Exception:
#     pass


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
        if cls_name in _CACHE_CLASS_NAMES:
            # 只清 outputs 缓存（key_class=CacheKeySetInputSignature）。
            # objects 缓存（CacheKeySetID）里是节点实例本身（含内部状态，例如
            # 缓存好的 LoRA），与定向清理的语义保持一致，故意不动。
            key_cls = getattr(getattr(obj, "key_class", None), "__name__", "")
            if key_cls != "CacheKeySetInputSignature":
                continue
            n = _clear_all_caches_recursive(obj, 0)
            cleared += n
    return cleared


# ===== RIFE 清理函数已停用，整体注释保留备查 =====
# def _clear_vfi_model_caches(which: str = "all") -> Dict[str, int]:
#     """Clear RIFE module-level caches.
#
#     Moves tensors to CPU first to avoid PyTorch C++ warning storms.
#
#     Args:
#         which: "all", "grid" (backwarp_tenGrid only), or "model" (_model_cache only)
#     """
#     result: Dict[str, int] = {}
#
#     if which in ("all", "model"):
#         if _RIFE_CACHE is not None:
#             try:
#                 n = 0
#                 for _key, model in list(_RIFE_CACHE.items()):
#                     try:
#                         model.cpu()
#                     except Exception:
#                         pass
#                     n += 1
#                 _RIFE_CACHE.clear()
#                 result["RIFE _model_cache"] = n
#             except Exception:
#                 pass
#
#     if which in ("all", "grid"):
#         if _BACKWARP_GRID is not None:
#             try:
#                 n = 0
#                 for _key, t in list(_BACKWARP_GRID.items()):
#                     try:
#                         if torch.is_tensor(t) and t.is_cuda:
#                             t.detach_()
#                             t.data = torch.empty(0, device="cpu")
#                     except Exception:
#                         pass
#                     n += 1
#                 _BACKWARP_GRID.clear()
#                 result["RIFE backwarp_tenGrid"] = n
#             except Exception:
#                 pass
#
#     for k, v in result.items():
#         log.debug("[DeepVRAMCleanup] Cleared %s: %d entries", k, v)
#
#     return result


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

# ===== 显存 -> 内存 offload 已停用 =====
# 用户明确不要这个功能：本节点的目的在腾内存，而它恰恰是把显存搬进内存（增加内存
# 占用）；且 ComfyUI 自身在显存吃紧时 free_memory -> ModelPatcher.partially_unload
# 已经做得很好。输入开关与 execute() 调用点均已移除。
# 如需恢复：取消本段注释，并在 define_schema 加回 offload_models_to_ram 开关、
# 在 execute() 里恢复 offload_lines 分支。
#
# def _offload_loaded_models_to_ram() -> List[str]:
#     """Move loaded model weights from VRAM to system RAM, keeping the models loaded."""
#     gb = 1024 ** 3
#     lines: List[str] = []
#     for loaded in list(mm.current_loaded_models):
#         patcher = loaded.model
#         if patcher is None:
#             continue
#         resident = patcher.loaded_size()
#         if resident <= 0:
#             continue
#         freed = patcher.partially_unload(patcher.offload_device, 1e32)
#         name = patcher.model.__class__.__name__
#         lines.append(f"offloaded {name}: {freed / gb:.3f} GB VRAM -> {patcher.offload_device} (was {resident / gb:.3f} GB)")
#         log.debug("[DeepVRAMCleanup] Offloaded %s: %.1f MB VRAM -> %s", name, freed / (1024 ** 2), patcher.offload_device)
#     return lines


# ---------------------------------------------------------------------------
#  Selective model unload (只卸载指定模型)
# ---------------------------------------------------------------------------

def _fallback_unload_model_and_clones(patcher, all_devices: bool = False) -> None:
    """内置回退实现，语义与内核 unload_model_and_clones 一致：白名单之外才允许卸载。

    keep_loaded 里保留除目标（及其同 clone_base_uuid 的 clone）以外的所有已加载模型，
    因此 free_memory 只会把目标卸掉。
    """
    target_uuid = getattr(patcher, "clone_base_uuid", None)
    keep_loaded = []
    for loaded in list(mm.current_loaded_models):
        p = loaded.model
        if p is None:
            continue
        if target_uuid is not None and getattr(p, "clone_base_uuid", None) == target_uuid:
            continue
        keep_loaded.append(loaded)

    devices = mm.get_all_torch_devices() if all_devices else [mm.get_torch_device()]
    for dev in devices:
        mm.free_memory(1e30, dev, keep_loaded)


def _selective_unload_one(patcher, unload_additional_models: bool, all_devices: bool) -> None:
    """卸载单个 ModelPatcher（含其 clone / 附属模型），其余已加载模型不受影响。"""
    if _unload_model_and_clones is not None:
        try:
            _unload_model_and_clones(
                patcher,
                unload_additional_models=unload_additional_models,
                all_devices=all_devices,
            )
            return
        except TypeError as exc:
            log.warning(
                "[DeepVRAMCleanup] unload_model_and_clones 调用签名不兼容(%s)，改用内置回退实现", exc
            )
    _fallback_unload_model_and_clones(patcher, all_devices=all_devices)


def _target_loaded_models(patcher) -> List[Any]:
    """内核眼里仍登记为「已加载」的目标模型（含其同 clone_base_uuid 的 clone）。"""
    target_uuid = getattr(patcher, "clone_base_uuid", None)
    if target_uuid is None:
        return []
    return [
        lm for lm in list(mm.current_loaded_models)
        if lm.model is not None and getattr(lm.model, "clone_base_uuid", None) == target_uuid
    ]


def _free_leftovers(patcher, all_devices: bool) -> List[str]:
    """按目标自身所在设备补刀。

    必需原因（已在 model_management.py:903 / 215-245 读码 + 本机实测确认）：
    free_memory 只处理 `shift_model.device == device` 的模型，而 unload_all_models /
    get_all_torch_devices 在有 GPU 时只返回 GPU 设备——于是 load_device 为 cpu 的
    LoadedModel（低显存下 text_encoder 的常见落点）永远不会被官方路径释放。

    这里对残留目标按其自身设备再发一次 free_memory，keep_loaded 排除目标组，其余模型原位保留。
    目标在其它 GPU 设备上且未开 all_devices 时不动手，只如实上报。
    """
    lines: List[str] = []
    leftovers = _target_loaded_models(patcher)
    if not leftovers:
        return lines

    devices: List[Any] = []
    for lm in leftovers:
        dev = getattr(lm, "device", None)
        if dev is None:
            continue
        is_host = getattr(dev, "type", None) == "cpu"
        if (is_host or all_devices) and dev not in devices:
            devices.append(dev)

    target_uuid = getattr(patcher, "clone_base_uuid", None)
    for dev in devices:
        keep_loaded = [
            lm for lm in list(mm.current_loaded_models)
            if lm.model is None or getattr(lm.model, "clone_base_uuid", None) != target_uuid
        ]
        try:
            freed = mm.free_memory(1e30, dev, keep_loaded)
        except Exception as exc:
            log.error(
                "[DeepVRAMCleanup] 在 %s 上补卸目标失败: %s", dev, exc, exc_info=True
            )
            lines.append(f"sweep FAILED on {dev}: {type(exc).__name__}: {exc}")
            continue
        log.debug("[DeepVRAMCleanup] 补卸 %s 上的目标模型 %d 个", dev, len(freed))
        lines.append(f"swept {len(freed)} on {dev}")

    remaining = _target_loaded_models(patcher)
    if remaining:
        devs = sorted({str(getattr(lm, "device", "?")) for lm in remaining})
        lines.append(f"still loaded: {devs}（已尝试所有设备，仍未释放，请看日志里的报错）")
        log.warning("[DeepVRAMCleanup] 目标模型仍驻留在 %s", devs)
    return lines


def _trim_working_set() -> Tuple[bool, str]:
    """EmptyWorkingSet：强制修剪工作集，促使堆把"已 free 但未去提交"的大块落地。

    背景（PersistentOutput 事故实测，2026-10-14）：Windows 堆对已 free 的脏大块
    惰性归还——引用清干净后 RSS 不降，已提交（commit）更是只有在堆真正去提交时
    才会回落。修剪工作集把驻留页挪去 standby（使用中/立即可见下降），模型页软缺
    页恢复（RAM 速度）。注意：commit 是否回落取决于 torch 分配层的去提交行为，
    修剪本身不保证。
    """
    try:
        import psutil
        gb = 1024 ** 3
        proc = psutil.Process()
        rss_before = proc.memory_info().rss
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        k32.SetProcessWorkingSetSize.restype = ctypes.c_int
        ok = k32.SetProcessWorkingSetSize(ctypes.c_void_p(-1), ctypes.c_size_t(-1), ctypes.c_size_t(-1))
        err = ctypes.get_last_error()
        rss_after = proc.memory_info().rss
        line = (f"EmptyWorkingSet ok={ok} err={err}: "
                f"RSS {rss_before / gb:.2f}GB -> {rss_after / gb:.2f}GB")
        if ok != 1:
            log.warning("[DeepVRAMCleanup] %s", line)
        return ok == 1, line
    except Exception as exc:
        return False, f"EmptyWorkingSet FAILED: {type(exc).__name__}: {exc}"


def _sysmem_line() -> str:
    """系统内存一行速览（RSS + 已提交），与任务管理器口径对照。"""
    try:
        import psutil
        gb = 1024 ** 3
        mi = psutil.Process().memory_info()
        return f"sysRAM: RSS={mi.rss / gb:.2f}GB COMMIT={mi.vms / gb:.2f}GB"
    except Exception:
        return "sysRAM: (psutil 不可用)"


def _describe_patcher(patcher) -> str:
    """给日志/诊断一个人能读的模型标识（类名 + 是否 text encoder + 加载设备）。"""
    try:
        inner = getattr(patcher, "model", None)
        cls = type(inner).__name__ if inner is not None else "?"
        kind = "text_encoder" if getattr(patcher, "is_clip", False) else "model"
        return f"{cls}[{kind}]@{(getattr(patcher, 'load_device', None))}"
    except Exception as exc:
        return f"<describe failed: {exc}>"


def _as_model_patcher(obj, label: str, lines: List[str]):
    """从节点端口对象取出 ModelPatcher。

    MODEL 端口传进来的本身就是 ModelPatcher；CLIP / VAE 是包装对象，patcher 挂在 .patcher 上。
    取不到就记日志 + 写诊断，绝不静默跳过。
    """
    if obj is None:
        return None
    try:
        patcher = getattr(obj, "patcher", obj)
    except Exception:
        patcher = None
    if patcher is None:
        msg = f"skip {label}: {type(obj).__name__} 上找不到 patcher（可能未加载权重）"
        log.warning("[DeepVRAMCleanup] %s", msg)
        lines.append(msg)
        return None
    if not hasattr(patcher, "clone_base_uuid") or not hasattr(patcher, "loaded_size"):
        msg = f"skip {label}: {type(patcher).__name__} 不是可识别的 ModelPatcher"
        log.warning("[DeepVRAMCleanup] %s", msg)
        lines.append(msg)
        return None
    return patcher


def _collect_model_targets(models: dict = None, clips: dict = None, vaes: dict = None) -> List[Tuple[str, Any]]:
    """把三组 autogrow dict（只含已连接项）整理成有序的 (标签, 端口对象) 列表。

    键是 autogrow 展开名（model0 / models.model0 皆可能），标签里带上方便 diag 阅读与排重。
    """
    targets: List[Tuple[str, Any]] = []
    for kind, values in (("MODEL", models or {}), ("CLIP", clips or {}), ("VAE", vaes or {})):
        for key in sorted(values.keys(), key=str):
            if values[key] is not None:
                targets.append((f"{kind}[{key}]", values[key]))
    return targets


def _unload_selected_models(
    targets: List[Tuple[str, Any]],
    unload_additional_models: bool,
    all_devices: bool,
) -> List[str]:
    """按 (标签, 端口对象) 列表逐个选择性卸载，返回诊断行。"""
    gb = 1024 ** 3
    lines: List[str] = []
    for label, obj in targets:
        patcher = _as_model_patcher(obj, label, lines)
        if patcher is None:
            continue
        name = _describe_patcher(patcher)
        before = _target_loaded_models(patcher)
        devs = sorted({str(getattr(lm, "device", "?")) for lm in before})
        try:
            resident = patcher.loaded_size()
        except Exception as exc:
            resident = -1
            log.warning("[DeepVRAMCleanup] 读取 %s 的 loaded_size 失败: %s", name, exc)

        try:
            _selective_unload_one(patcher, unload_additional_models, all_devices)
        except Exception as exc:
            log.error(
                "[DeepVRAMCleanup] 卸载 %s(%s) 失败: %s", label, name, exc, exc_info=True
            )
            lines.append(f"FAILED unload {label} {name}: {type(exc).__name__}: {exc}")
            continue

        # 官方路径只覆盖 GPU 设备，主机内存常驻的目标要靠这一步补卸
        lines.extend(_free_leftovers(patcher, all_devices))

        remaining = _target_loaded_models(patcher)
        if not before:
            lines.append(f"{label} {name}: 内核未登记为已加载，无需释放")
            log.debug("[DeepVRAMCleanup] %s %s: 未驻留", label, name)
        elif remaining:
            still = sorted({str(getattr(lm, "device", "?")) for lm in remaining})
            lines.append(f"partial {label} {name}: 已处理，但仍驻留 {still}")
        else:
            host_note = "（主机内存常驻：已解下登记与 patch，基础权重待引用释放后才真正归还）" if "cpu" in devs else ""
            lines.append(f"unloaded {label} {name}: {max(resident, 0) / gb:.3f} GB @ {','.join(devs)}{host_note}")
            log.debug(
                "[DeepVRAMCleanup] Unloaded %s %s: %.1f MB @ %s", label, name, max(resident, 0) / (1024 ** 2), devs
            )
    return lines


# ---------------------------------------------------------------------------
#  Per-node execution-cache eviction (只清指定节点的运行缓存)
# ---------------------------------------------------------------------------
#
#  依据（均在本机 ComfyUI 源码中读码确认）：
#   * 节点输出存在 caches.outputs，值为 CacheEntry(ui, outputs)（execution.py:109 / :618），
#     outputs 里就是该节点的输出对象（含 tensor 与 ModelPatcher）；
#   * 该缓存的 key 按节点算出：cache.cache_key_set.get_data_key(node_id)（caching.py:206 等）；
#   * 层次缓存要先下钻到真正持有条目的子缓存：_get_cache_for(node_id)（caching.py:366）；
#   * LRU / RAM 压力缓存还有同 key 的 used_generation / timestamps / children 字典，
#     内核自己删条目时是一起删的（caching.py:460-463 / :611-614），手动删必须镜像；
#   * 节点执行时拿到的是解析后的“值”，链接只存在于 prompt 里 —— 所以上游节点号必须
#     从 hidden 输入 PROMPT 里按自己的 unique_id 读回（execution.py:219 / :227）；
#   * ExecutionList.output_cache 就是活着的 caches.outputs（execution.py:779），
#     而 ExecutionList 由 hidden 输入 EXECUTION_LIST 交给节点（execution.py:223）。
#
#  安全性判据（读码推导，避免“清了又活”或“下游拿到 None”）：
#   下游只在 ExecutionList.get_cache（graph.py:244）里读边缓存，而边缓存持有同一个
#   CacheEntry（cache_link/cache_update，graph.py:233-260），并在命中时写回主缓存。
#   因此只要还有未执行的下游消费者，清主缓存要么被写回（引用没断），要么让它重新
#   入队重算（两个消费者可能拿到不同的值）。所以：只有在目标节点没有其它待执行
#   消费者时才真正动手，否则跳过并如实上报。

_CACHE_CLASS_NAMES = ("BasicCache", "HierarchicalCache", "LRUCache", "RAMPressureCache")


def _is_node_link(value) -> bool:
    """判断一个输入值是不是节点连线 [node_id, output_index]。"""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


# autogrow 展开后的 source 组键：source0..4 或 cache_sources.source0..4（序数一律取尾部）
_SOURCE_KEY_RE = re.compile(r"(^|\.)source\d+$")


def _linked_source_node_ids(unique_id, prompt) -> Tuple[List[str], str]:
    """从 prompt 图里读回 source 组全部端口连着的上游节点（去重、保序）。

    返回 (node_id 列表, 失败原因)。节点执行时只能拿到解析后的值，
    链接本身只在 prompt 里；autogrow 组的键名跨前端/后端有两种形态
    （sourceN / cache_sources.sourceN），用正则统一匹配。
    """
    if unique_id is None:
        return [], "拿不到本节点 id（hidden UNIQUE_ID 缺失）"
    if not isinstance(prompt, dict) or not prompt:
        return [], "拿不到 prompt 图（hidden PROMPT 缺失）"
    node = prompt.get(str(unique_id))
    if node is None:
        node = prompt.get(unique_id)
    if not isinstance(node, dict):
        return [], f"prompt 图里找不到本节点 #{unique_id}"
    ids: List[str] = []
    for key, value in (node.get("inputs") or {}).items():
        if not _SOURCE_KEY_RE.search(key):
            continue
        if _is_node_link(value):
            sid = str(value[0])
            if sid not in ids:
                ids.append(sid)
    return ids, ("" if ids else "source 端口没有连线")


def _node_consumers(node_id, prompt) -> List[str]:
    """prompt 图里所有消费 node_id 输出的节点（不含动态子图里临时生成的节点）。"""
    consumers: List[str] = []
    if not isinstance(prompt, dict):
        return consumers
    target = str(node_id)
    known = {str(k) for k in prompt.keys()}
    for nid, node in prompt.items():
        inputs = (node or {}).get("inputs") or {}
        for value in inputs.values():
            if _is_node_link(value) and str(value[0]) == target and str(value[0]) in known:
                consumers.append(str(nid))
                break
    return consumers


def _pending_node_ids(execution_list):
    """当前还没执行的节点集合；拿不到返回 None（调用方按保守策略处理）。"""
    pending = getattr(execution_list, "pendingNodes", None)
    if isinstance(pending, dict):
        return {str(k) for k in pending.keys()}
    return None


def _size_cache_entry(entry) -> float:
    """估算缓存条目里 tensor 的字节数（同一 storage 只算一次）。"""
    seen = set()
    total = 0

    def walk(value, depth=0):
        nonlocal total
        if value is None or depth > 4:
            return
        if torch.is_tensor(value):
            try:
                storage = value.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            except Exception:
                pass
            return
        if isinstance(value, dict):
            value = list(value.values())
        elif not isinstance(value, (list, tuple)):
            return
        for item in value:
            walk(item, depth + 1)

    try:
        walk(getattr(entry, "outputs", None))
    except Exception as exc:
        log.warning("[DeepVRAMCleanup] 估算缓存条目体积失败: %s", exc)
    return float(total)


def _cache_objects_for_eviction(execution_list) -> List[Any]:
    """拿到输出缓存对象：优先 ExecutionList.output_cache（活的 caches.outputs），否则 gc 扫描回退。"""
    cache = getattr(execution_list, "output_cache", None) if execution_list is not None else None
    if cache is not None:
        return [cache]
    found = []
    for obj in gc.get_objects():
        if type(obj).__name__ in _CACHE_CLASS_NAMES:
            found.append(obj)
    return found


def _lookup_cache_entry(cache, node_id):
    """定位真正持有 node_id 条目的缓存对象与 key（层次缓存需下钻子缓存）。"""
    target = cache
    resolve = getattr(cache, "_get_cache_for", None)
    if callable(resolve):
        try:
            target = resolve(node_id)
        except Exception as exc:
            log.warning(
                "[DeepVRAMCleanup] 在 %s 上定位子缓存失败: %s", type(cache).__name__, exc, exc_info=True
            )
            return None, None
    if target is None:
        return None, None
    key_set = getattr(target, "cache_key_set", None)
    if key_set is None or not hasattr(key_set, "get_data_key"):
        return None, None
    key = key_set.get_data_key(node_id)
    if key is None:
        return None, None
    return target, key


def _measure_node_output_cache(node_id, execution_list) -> str:
    """只测不清理：报告 node_id 当前输出缓存条目数与 tensor 体量（dry-run）。"""
    caches = _cache_objects_for_eviction(execution_list)
    if not caches:
        return f"node {node_id}: 找不到输出缓存对象"
    found = 0
    freed = 0.0
    for cache in caches:
        target, key = _lookup_cache_entry(cache, node_id)
        if target is None or key is None:
            continue
        store = getattr(target, "cache", None)
        if not isinstance(store, dict) or key not in store:
            continue
        found += 1
        freed += _size_cache_entry(store[key])
    if found == 0:
        return f"node {node_id}: 输出缓存里没有条目（未执行或已被释放）"
    return f"node {node_id}: 可释放 {found} 个缓存条目, ~{freed / (1024 ** 2):.1f} MB tensor"


def _drop_edge_refs(execution_list, node_id) -> int:
    """断开 ExecutionList 边缓存里对目标条目的引用。

    不清这里的话，下游 get_cache 命中时会把它写回主缓存（graph.py:244-252），
    等于“清了又活”、引用也没真正断开。
    """
    dropped = 0
    edge = getattr(execution_list, "execution_cache", None)
    if isinstance(edge, dict):
        if edge.pop(node_id, None) is not None:
            dropped += 1
        for mapping in edge.values():
            if isinstance(mapping, dict) and mapping.pop(node_id, None) is not None:
                dropped += 1
    listeners = getattr(execution_list, "execution_cache_listeners", None)
    if isinstance(listeners, dict):
        listeners.pop(node_id, None)
    return dropped


def _evict_node_output_cache(node_id, execution_list, our_id, prompt) -> Tuple[List[str], int]:
    """清掉 node_id 在输出缓存里的条目，返回 (诊断行, 实际删除的条目数)。"""
    lines: List[str] = []
    removed_total = 0
    caches = _cache_objects_for_eviction(execution_list)
    if not caches:
        msg = f"无法清理节点 {node_id}：找不到输出缓存对象（execution_list 与 gc 扫描都没有）"
        log.error("[DeepVRAMCleanup] %s", msg)
        lines.append(msg)
        return lines, 0

    # 安全判据：目标还有别的待执行消费者时不动手
    consumers = [c for c in _node_consumers(node_id, prompt) if our_id is None or c != str(our_id)]
    pending = _pending_node_ids(execution_list)
    if pending is None:
        blockers = consumers
        reason = "无法判定调度状态，保守跳过"
    else:
        blockers = [c for c in consumers if c in pending]
        reason = f"仍有下游待执行: {sorted(blockers)}"
    if blockers:
        msg = f"skip clear node {node_id}: {reason}（清了会被写回或让它们重算，可能取到不同结果）"
        log.warning("[DeepVRAMCleanup] %s", msg)
        lines.append(msg)
        return lines, 0

    removed = 0
    freed = 0.0
    for cache in caches:
        target, key = _lookup_cache_entry(cache, node_id)
        if target is None or key is None:
            continue
        store = getattr(target, "cache", None)
        if not isinstance(store, dict) or key not in store:
            continue
        entry = store[key]
        freed += _size_cache_entry(entry)
        try:
            del store[key]
        except Exception as exc:
            log.error(
                "[DeepVRAMCleanup] 删除 %s 中的节点 %s 缓存条目失败: %s",
                type(cache).__name__, node_id, exc, exc_info=True,
            )
            lines.append(f"FAILED clear node {node_id} in {type(cache).__name__}: {type(exc).__name__}: {exc}")
            continue
        # 镜像内核自己的删除动作，避免 side dict 残留导致 KeyError
        for side in ("used_generation", "timestamps", "children"):
            side_dict = getattr(target, side, None)
            if isinstance(side_dict, dict):
                side_dict.pop(key, None)
        removed += 1

    edge_dropped = _drop_edge_refs(execution_list, node_id)
    if removed == 0:
        msg = f"node {node_id}: 输出缓存里没有条目（未执行或已被释放）"
        lines.append(msg)
        log.debug("[DeepVRAMCleanup] %s", msg)
    else:
        msg = (
            f"cleared node {node_id}: {removed} 个缓存条目, "
            f"{freed / (1024 ** 2):.1f} MB tensor, 边引用 {edge_dropped} 处"
        )
        lines.append(msg)
        log.debug("[DeepVRAMCleanup] %s", msg)
    return lines, removed


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

class DeepVRAMCleanup(io.ComfyNode):
    """Selective memory/VRAM cleanup with per-source toggles.

    开关按「作用对象 + 全局/定向」成对排列：
    --------------------  --  ----------------------------------------------
    第一组 运行缓存
    clear_all_node_cache    清空所有中间节点输出缓存（全局）
    clear_source_node_cache 只清 source 组可增长端口连着的那些节点的缓存条目
    第二组 模型
    unload_models            卸载所有已加载模型（全局）
    unload_connected_models  只卸 models/clips/vaes 可增长端口连进来的全部模型
    第三组 收尾释放
    gc_empty_cache           gc.collect() + torch.cuda.empty_cache()
    trim_cuda_pool           Direct cudaMemPoolTrimTo(0) (cudaMallocAsync)
    purge_orphan_tensors     Hunt & detach stray CUDA tensors from heap
    --------------------  --  ----------------------------------------------
    输入组是 io.Autogrow 可增长列表（同 AnySwitch）：每组初始 1 个口，
    连满自动长出下一个（models/clips/vaes/cache_sources 各最多 5 个），
    执行侧 dict 只含已连接项。
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DeepVRAMCleanup",
            display_name="Deep VRAM Cleanup",
            category="utils/VRAM",
            # 输出节点：允许在画布上直接点节点的运行按钮单独执行本节点
            # （前端只把被点名的节点作为 partial_execution_targets 发给后端）。
            is_output_node=True,
            inputs=[
                # ===== 第一组：运行缓存（全局清 -> 只清连线节点）=====
                io.Boolean.Input("clear_all_node_cache", default=True,
                    tooltip="【清缓存 · 全局】清空 ComfyUI 所有节点的中间输出缓存。每个节点跑完后的输出会被缓存以加速下次执行，但也因此持有大量中间 tensor 的引用。此开关清除所有已缓存的中间结果，不影响已加载的模型。\n\n这是腾内存的主力（中间结果占的往往是内存里的大头）。代价：清掉后所有分支都得重算，缓存加速失效。\n\n与下面 clear_source_node_cache（只清连线指定的那一个节点）是一对：全清粗暴但彻底，定向清理精细但有前提。"),
                io.Boolean.Input("clear_source_node_cache", default=True,
                    tooltip="【清缓存 · 只清指定的】清掉 source 组可增长端口连着的那些节点在「运行缓存」里的输出条目（只清它们，不动其它节点；可以接多个，逐一清理）。\n\n用途：把某些节点的中间结果（比如一整张 IMAGE、latent）从缓存里摘出来，解除它们占用的内存/显存引用；代价是下次需要这些输出时对应节点会重新执行。比上面的全局清理精细得多。\n\n注意：全局清理 clear_all_node_cache 开启时本开关会被跳过（全清已覆盖）。\n\n安全性（已读码确认）：只在目标节点没有「其它待执行的下游消费者」时才真正删条目。否则跳过并在 diag 里说明——因为下游是从边缓存读取的，删主缓存要么被它写回（引用没断），要么让它重新入队重算（两个消费者可能拿到不同的值）。\n\n注意：只作用于 outputs 缓存（节点输出）。objects 缓存里放的是节点实例本身（含它自己的内部状态，例如缓存好的 LoRA），故意不动。\n\n未连线时本开关不做事。"),
                # ===== 第二组：模型（全局卸 -> 只卸连线模型）=====
                io.Boolean.Input("unload_models", default=False,
                    tooltip="【卸模型 · 全局】卸载 ComfyUI 模型管理中所有已加载的模型（checkpoint / LoRA / VAE 等）。清理后再次生成图像需要重新加载模型，会有短暂延迟。日常不建议开启，除非内存/显存极度紧张。"),
                io.Boolean.Input("unload_connected_models", default=False,
                    tooltip="【卸模型 · 只卸指定的】只卸载右侧 model / clip / vae 三个可选端口连进来的那个模型，其余已加载模型保持不动。\n\n走 ComfyUI 内核的选择性卸载接口（comfy.model_management.unload_model_and_clones），按 clone_base_uuid 把目标模型本身、它的 clone（如 CLIPSetLastLayer 产生的副本、LoRA patch 过的副本）以及挂在它上面的附属模型一起卸掉（附属模型一定连带，所有设备上的残留也一并清理）。\n\n注意：必须至少连一个 model / clip / vae 端口才有效果，三者都不连时本开关不做事（会在 diag 里说明）。同时开启 unload_models 时以全卸为准，本开关自动跳过。\n\n前提：本节点必须位于通往「输出节点」的链路上才会被执行——用 passthrough 端口把它串在流程中间，或把它的输出连到后续节点，否则 ComfyUI 会直接跳过这个节点。\n\n典型用法：采样跑完后只卸 text_encoder（几 GB）给 VAE 解码腾地方，同时保留 checkpoint 不重载。"),
                # ===== 第三组：收尾释放（显存池 / 堆里的残留引用）=====
                io.Boolean.Input("gc_empty_cache", default=True,
                    tooltip="【收尾】执行 Python gc.collect() + torch.cuda.empty_cache()。释放 Python 层不再被引用的对象，并将 PyTorch CUDA 缓存池中已标记为空闲的内存归还给操作系统。"),
                io.Boolean.Input("trim_cuda_pool", default=True,
                    tooltip="【收尾】直接调用 CUDA Runtime 的 cudaMemPoolTrimTo(0)，强制将 cudaMallocAsync 内存池中所有未使用的内存归还给操作系统。\n\n这是对 torch.cuda.empty_cache() 的补充——在 CUDA 13.x + cudaMallocAsync 后端下，仅靠 empty_cache() 往往无法彻底释放 reserved 内存。此选项直接绕过 PyTorch 操作 CUDA 内存池。"),
                io.Boolean.Input("purge_orphan_tensors", default=False,
                    tooltip="【收尾 · 暴力模式】遍历 Python 堆中所有 gc 追踪的对象，找出散落在类属性、全局变量等角落的 GPU tensor，强制 detach 或移动到 CPU。\n\n此操作慢且激进，可能影响其他节点的内部状态。仅在以上清理都无效时作为最后手段使用。"),
                # ===== 以下开关已按用户要求从 UI 移除（语义固定 / 停用），详见文件顶部注释：
                #   offload_models_to_ram     —— 停用：目的在腾内存，而它把显存搬进内存（增加内存占用）
                #   unload_additional_models  —— 固定 True（连带卸载附属模型）
                #   unload_all_devices        —— 固定 True（所有设备上的残留一起清）
                #   rife_backwarp_grid / rife_model_cache —— 移除：不再使用插帧节点
                io.AnyType.Input("passthrough1", optional=True),
                io.AnyType.Input("passthrough2", optional=True),
                io.AnyType.Input("passthrough3", optional=True),
                io.AnyType.Input("passthrough4", optional=True),
                # ===== 可增长输入组（io.Autogrow，同 AnySwitch 机制：连满自动长、断开自动收）=====
                # min=0：初始每组 1 个口，全部按 optional 展开；执行侧 dict 只含已连接项。
                io.Autogrow.Input(
                    "models",
                    template=io.Autogrow.TemplatePrefix(
                        io.Model.Input("model"), prefix="model", min=0, max=5),
                    optional=True,
                    tooltip="可增长的 diffusion 模型输入（model0…model4）。接进来的每一个都会被单独卸载，其它已加载模型不动。配合 unload_connected_models 开关使用。",
                ),
                io.Autogrow.Input(
                    "clips",
                    template=io.Autogrow.TemplatePrefix(
                        io.Clip.Input("clip"), prefix="clip", min=0, max=5),
                    optional=True,
                    tooltip="可增长的 text encoder 输入（clip0…clip4）。接进来的每一个都会被单独卸载，不碰 diffusion 模型和 VAE。显存不紧张时 TE 常驻在系统内存——本节点会按目标自身所在设备补卸。",
                ),
                io.Autogrow.Input(
                    "vaes",
                    template=io.Autogrow.TemplatePrefix(
                        io.Vae.Input("vae"), prefix="vae", min=0, max=5),
                    optional=True,
                    tooltip="可增长的 VAE 输入（vae0…vae4）。接进来的每一个都会被单独卸载。VAE 体积小、重载快，跑完大分辨率解码后卸载是纯赚的。",
                ),
                io.Autogrow.Input(
                    "cache_sources",
                    template=io.Autogrow.TemplatePrefix(
                        io.AnyType.Input("source"), prefix="source", min=0, max=5),
                    optional=True,
                    tooltip="可增长的「要清运行缓存的节点」输入（source0…source4）：把要清的节点的任意输出接进来，可以接多个，逐一清理。\n\n节点执行时只能拿到解析后的值，链接本身只存在于 prompt 图里——所以上游节点号是从隐藏输入 PROMPT 里按本节点 id 读回来的（同一个上游接多个口会自动去重）。配合 clear_source_node_cache 开关使用。",
                ),
            ],
            outputs=[
                io.AnyType.Output("passthrough1", display_name="passthrough1"),
                io.AnyType.Output("passthrough2", display_name="passthrough2"),
                io.AnyType.Output("passthrough3", display_name="passthrough3"),
                io.AnyType.Output("passthrough4", display_name="passthrough4"),
                io.Float.Output("used_gb", display_name="used_gb"),
                io.Float.Output("total_gb", display_name="total_gb"),
                io.Float.Output("freed_gb", display_name="freed_gb"),
                io.String.Output("diag", display_name="diag"),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.execution_list],
        )

    @classmethod
    def _vram(cls, device: torch.device) -> Tuple[float, float]:
        if device.type != "cuda":
            return 0.0, 0.0
        gb = 1024**3
        a = torch.cuda.memory_allocated(device) / gb
        r = torch.cuda.memory_reserved(device) / gb
        return max(a, r), torch.cuda.get_device_properties(device).total_memory / gb

    @classmethod
    def execute(
        cls,
        # 第一组：运行缓存（全局 -> 指定节点）
        clear_all_node_cache: bool,
        clear_source_node_cache: bool,
        # 第二组：模型（全局 -> 指定模型）
        unload_models: bool,
        unload_connected_models: bool,
        # 第三组：收尾释放
        gc_empty_cache: bool,
        trim_cuda_pool: bool,
        purge_orphan_tensors: bool,
        passthrough1=None,
        passthrough2=None,
        passthrough3=None,
        passthrough4=None,
        models: dict = None,
        clips: dict = None,
        vaes: dict = None,
        cache_sources: dict = None,
    ):
        # hidden 输入（unique_id / prompt / execution_list）由 V3 框架经
        # PREPARE_CLASS_CLONE 挂到 cls.hidden 上，不再是 execute 的形参。
        device = mm.get_torch_device()
        used_before, total = cls._vram(device)
        sysmem_before = _sysmem_line()

        # ===================== 第一组：运行缓存 =====================
        # 先丢引用（清缓存），再统一释放（第三组），这样释放才真的落到刚丢掉的 tensor 上。
        need_trim = False
        if clear_all_node_cache:
            n = _find_and_clear_execution_caches()
            log.debug("[DeepVRAMCleanup] Cleared %d execution-cache entries", n)
            need_trim = need_trim or n > 0

        # 指定节点的运行缓存（只清 source 组端口连着的那些节点，可多个、自动去重）
        cache_lines: List[str] = []
        if clear_source_node_cache:
            if clear_all_node_cache:
                msg = "skip 指定节点缓存清理: clear_all_node_cache 已开启，全清已覆盖"
                log.debug("[DeepVRAMCleanup] %s", msg)
                cache_lines.append(msg)
            else:
                hidden = cls.hidden
                source_ids, why = _linked_source_node_ids(hidden.unique_id, hidden.prompt)
                if not source_ids:
                    msg = f"指定节点缓存清理已开启，但没找到目标节点：{why}"
                    log.warning("[DeepVRAMCleanup] %s", msg)
                    cache_lines.append(msg)
                else:
                    log.debug(
                        "[DeepVRAMCleanup] 准备清理 %d 个节点的运行缓存（由 source 连线指定）: %s",
                        len(source_ids), source_ids,
                    )
                    for source_id in source_ids:
                        lines, removed = _evict_node_output_cache(
                            source_id, hidden.execution_list, hidden.unique_id, hidden.prompt
                        )
                        cache_lines.extend(lines)
                        need_trim = need_trim or removed > 0
        elif cache_sources:
            # 干跑模式：只统计各 source 节点当前可释放的缓存量，不动任何条目。
            # 结果进 diag（前端上屏），供决定是否开启 clear_source_node_cache。
            hidden = cls.hidden
            source_ids, why = _linked_source_node_ids(hidden.unique_id, hidden.prompt)
            if not source_ids:
                msg = f"source 已连线但没找到目标节点：{why}"
                log.warning("[DeepVRAMCleanup] %s", msg)
                cache_lines.append(msg)
            else:
                cache_lines.append("dry-run 未开启清理，各 source 节点当前可释放的缓存：")
                for source_id in source_ids:
                    cache_lines.append(_measure_node_output_cache(
                        source_id, hidden.execution_list
                    ))

        # ===================== 第二组：模型 =====================
        # 指定卸载：只卸指定端口连进来的模型（可与全卸同时勾选，此时以全卸为准）
        # 附属模型与其它设备上的残留固定为一起清理（原来是对应开关，按用户要求收进内部常开）。
        unload_lines: List[str] = []
        if unload_connected_models:
            if unload_models:
                msg = "skip 指定卸载: unload_models 已开启，全卸覆盖本次选择性卸载"
                log.debug("[DeepVRAMCleanup] %s", msg)
                unload_lines.append(msg)
            else:
                targets = _collect_model_targets(models, clips, vaes)
                if not targets:
                    msg = "指定卸载已开启，但 model / clip / vae 组里没有任何连线，未卸载任何模型"
                    log.warning("[DeepVRAMCleanup] %s", msg)
                    unload_lines.append(msg)
                else:
                    unload_lines = _unload_selected_models(
                        targets,
                        unload_additional_models=True,
                        all_devices=True,
                    )
        elif any((models, clips, vaes)):
            log.debug(
                "[DeepVRAMCleanup] model/clip/vae 组已连线，但 unload_connected_models 未开启，保持不变"
            )

        if unload_models:
            cleanup_models()
            cleanup_models_gc()
            unload_all_models()
            need_trim = True

        # ===== 显存 -> 内存 offload 已停用（用户不要；ComfyUI 自身处理得很好）=====
        # offload_lines: List[str] = []
        # if offload_models_to_ram:
        #     offload_lines = _offload_loaded_models_to_ram()

        # ===== RIFE 缓存清理已停用（见文件顶部说明）=====
        # rife_which = (
        #     "all" if (rife_backwarp_grid and rife_model_cache)
        #     else "grid" if rife_backwarp_grid
        #     else "model" if rife_model_cache
        #     else ""
        # )
        # if rife_which:
        #     _clear_vfi_model_caches(which=rife_which)

        # ===================== 第三组：收尾释放 =====================
        if gc_empty_cache:
            gc.collect()
            if device.type == "cuda":
                soft_empty_cache(force=True)

        if purge_orphan_tensors:
            n = _purge_cuda_orphans()
            log.debug("[DeepVRAMCleanup] Purged %d orphan CUDA tensors", n)

        # Final GC + cache flush
        for _ in range(3):
            gc.collect()
        if device.type == "cuda":
            soft_empty_cache(force=True)

        if trim_cuda_pool and device.type == "cuda":
            _trim_cuda_mempool(device.index if hasattr(device, "index") else 0)

        # 系统内存归还：Windows 堆对已 free 的脏大块惰性归还（PersistentOutput
        # 事故实测）——引用清干净后 RSS 也不降，且每次重新分配都提交全新内存。
        # 只要本轮发生过成功的清理/卸载就强制修剪工作集，让归还确定性落地
        # （不再依赖 gc_empty_cache 开关——实测不修剪时第二次清理就会失效）。
        ws_line = ""
        if need_trim or gc_empty_cache:
            ws_ok, ws_line = _trim_working_set()
            if not ws_ok:
                log.warning("[DeepVRAMCleanup] %s", ws_line)

        # --- Diagnostics ---
        used_after, _ = cls._vram(device)
        freed = max(0.0, used_before - used_after)

        lines = [
            f"used: {used_before:.3f} -> {used_after:.3f} GB  (freed {freed:.3f})",
            f"total: {total:.2f} GB",
            f"sysRAM before: {sysmem_before}",
            _sysmem_line(),
            "",
        ]
        if ws_line:
            lines.append(ws_line)
            lines.append("")
        lines.extend(_inspect_comfy_caches())
        if unload_lines:
            lines.append("")
            lines.extend(unload_lines)
        if cache_lines:
            lines.append("")
            lines.extend(cache_lines)
        lines.append("")
        lines.extend(_find_gpu_tensors_direct())

        diag = "\n".join(lines)
        # diag 同时挂进 ui：前端扩展（ui/deep_vram_cleanup.js）把它渲染到节点上，
        # 命令行日志之外画布直接可见（清理/跳过原因一目了然）。
        return io.NodeOutput(
            passthrough1, passthrough2, passthrough3, passthrough4,
            used_after, total, freed, diag,
            ui={"diag": [diag]},
        )
