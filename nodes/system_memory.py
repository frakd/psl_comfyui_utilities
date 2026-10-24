"""Windows system-wide memory / GPU memory sampling.

RAM/commit come from GlobalMemoryStatusEx.  VRAM usage comes from the PDH
"GPU Adapter Memory" counters (system-wide, all processes); DXGI supplies the
NVIDIA adapter's LUID, name, capacity and budgets.  This avoids the known
inaccuracy of IDXGIAdapter3::QueryVideoMemoryInfo::CurrentUsage on hybrid
graphics setups, where CUDA context memory can be missing.

This module deliberately has no ComfyUI imports so it can be tested in a
standalone Python process.
"""

from __future__ import annotations

import ctypes
import logging
import os
import uuid
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("psl_comfyui_utilities.system_memory")


# ---------------------------------------------------------------------------
#  Win32 / DXGI / PDH definitions
# ---------------------------------------------------------------------------

class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, value: str):
        parsed = uuid.UUID(value)
        super().__init__(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            (ctypes.c_ubyte * 8)(*parsed.bytes[8:]),
        )


_IID_IDXGIFactory1 = _GUID("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_IDXGIAdapter3 = _GUID("645967A4-1392-4310-A798-8053CE3E93FD")
_DXGI_ERROR_NOT_FOUND = 0x887A0002

# IUnknown 0-2, IDXGIObject 3-6, IDXGIAdapter 7-9,
# IDXGIAdapter1.GetDesc1 = 10, IDXGIAdapter3.QueryVideoMemoryInfo = 14.
_VT_QUERY_INTERFACE = 0
_VT_RELEASE = 2
_VT_FACTORY_ENUM_ADAPTERS1 = 12
_VT_ADAPTER_GET_DESC1 = 10
_VT_ADAPTER3_QUERY_MEMORY_INFO = 14

_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
_DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1

_COINIT_MULTITHREADED = 0x0
_RPC_E_CHANGED_MODE = 0x80010106
_S_FALSE = 1


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [
        ("szName", ctypes.c_wchar_p),
        ("CStatus", ctypes.c_ulong),
        ("doubleValue", ctypes.c_double),
    ]


_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2


def _vtable_call(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])
    return fn


def _com_release(obj: Optional[ctypes.c_void_p]) -> None:
    if obj and obj.value:
        try:
            _vtable_call(obj, _VT_RELEASE, ctypes.c_ulong)(obj)
        except Exception:
            log.debug("COM Release failed", exc_info=True)


def _read_dxgi_info() -> Dict[str, Any]:
    """One-shot DXGI adapter discovery (LUID, name, capacity, budgets)."""
    info: Dict[str, Any] = {
        "gpu_name": None,
        "dedicated_total": None,
        "dedicated_budget": None,
        "shared_budget": None,
        "luid_low": None,
        "luid_high": None,
        "error": None,
    }
    ole32 = ctypes.WinDLL("ole32")
    dxgi = ctypes.WinDLL("dxgi.dll")
    com_initialized = False
    factory = adapter = adapter3 = None
    try:
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        ole32.CoInitializeEx.restype = ctypes.c_long
        hr = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
        unsigned = hr & 0xFFFFFFFF
        if hr == 0 or unsigned == _S_FALSE:
            com_initialized = True
        elif unsigned != _RPC_E_CHANGED_MODE:
            raise OSError(f"CoInitializeEx failed: 0x{unsigned:08x}")

        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        create_factory.restype = ctypes.c_long
        factory = ctypes.c_void_p()
        hr = create_factory(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
        if hr != 0 or not factory.value:
            raise OSError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08x}")

        enum_adapters = _vtable_call(
            factory, _VT_FACTORY_ENUM_ADAPTERS1, ctypes.c_long,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
        )

        best = None
        index = 0
        while True:
            current = ctypes.c_void_p()
            hr = enum_adapters(factory, index, ctypes.byref(current))
            if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND:
                break
            if hr != 0 or not current.value:
                break
            desc = _DXGI_ADAPTER_DESC1()
            get_desc1 = _vtable_call(
                current, _VT_ADAPTER_GET_DESC1, ctypes.c_long,
                ctypes.POINTER(_DXGI_ADAPTER_DESC1),
            )
            if get_desc1(current, ctypes.byref(desc)) != 0:
                _com_release(current)
                index += 1
                continue
            # Only the CUDA (NVIDIA) adapter is monitored; a larger
            # non-NVIDIA adapter must not win.
            if desc.VendorId == 0x10DE:
                score = int(desc.DedicatedVideoMemory)
                if best is None or score > best[0]:
                    if best is not None:
                        _com_release(best[1])
                    best = (score, current, desc)
                else:
                    _com_release(current)
            else:
                _com_release(current)
            index += 1

        if best is None:
            raise RuntimeError("no NVIDIA DXGI adapter found")
        _, adapter, desc = best
        info["gpu_name"] = str(desc.Description).rstrip("\x00")
        info["dedicated_total"] = int(desc.DedicatedVideoMemory)
        info["dedicated_budget"] = int(desc.DedicatedVideoMemory)
        info["shared_budget"] = int(desc.SharedSystemMemory)
        info["luid_low"] = int(desc.AdapterLuid.LowPart)
        info["luid_high"] = int(desc.AdapterLuid.HighPart)

        adapter3 = ctypes.c_void_p()
        query_interface = _vtable_call(
            adapter, _VT_QUERY_INTERFACE, ctypes.c_long,
            ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
        )
        hr = query_interface(adapter, ctypes.byref(_IID_IDXGIAdapter3), ctypes.byref(adapter3))
        if hr == 0 and adapter3.value:
            query = _vtable_call(
                adapter3, _VT_ADAPTER3_QUERY_MEMORY_INFO, ctypes.c_long,
                ctypes.c_uint, ctypes.c_int,
                ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
            )
            for group, key in (
                (_DXGI_MEMORY_SEGMENT_GROUP_LOCAL, "dedicated_budget"),
                (_DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, "shared_budget"),
            ):
                value = _DXGI_QUERY_VIDEO_MEMORY_INFO()
                if query(adapter3, 0, group, ctypes.byref(value)) == 0 and value.Budget:
                    info[key] = int(value.Budget)
        else:
            log.debug("[MemoryMonitor] IDXGIAdapter3 unavailable; using DXGI adapter capacities")
    except Exception as exc:
        info["error"] = f"DXGI 初始化失败: {type(exc).__name__}: {exc}"
        log.warning("[MemoryMonitor] %s", info["error"], exc_info=True)
    finally:
        _com_release(adapter3)
        _com_release(adapter)
        _com_release(factory)
        if com_initialized:
            try:
                ole32.CoUninitialize()
            except Exception:
                log.debug("CoUninitialize failed", exc_info=True)
    return info


class _PDHGPUMemory:
    """Per-sample dedicated/shared usage for one adapter LUID."""

    def __init__(self, luid_low: int, luid_high: int) -> None:
        self._pdh = ctypes.WinDLL("pdh.dll")
        self._query = ctypes.c_void_p()
        self._dedicated = ctypes.c_void_p()
        self._shared = ctypes.c_void_p()
        self._needle = f"0x{luid_high & 0xFFFFFFFF:08x}_0x{luid_low:08x}".lower()
        self.error: Optional[str] = None
        self._define_api()
        try:
            self._open()
        except Exception as exc:
            self.error = f"PDH 初始化失败: {type(exc).__name__}: {exc}"
            log.warning("[MemoryMonitor] %s", self.error, exc_info=True)
            self.close()

    def _define_api(self) -> None:
        pdh = self._pdh
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhOpenQueryW.restype = ctypes.c_long
        pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
        ]
        pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
        pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        pdh.PdhCollectQueryData.restype = ctypes.c_long
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p,
        ]
        pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_long
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        pdh.PdhCloseQuery.restype = ctypes.c_long

    def _open(self) -> None:
        hr = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if hr != 0:
            raise OSError(f"PdhOpenQueryW failed: 0x{hr & 0xFFFFFFFF:08x}")
        for path, target in (
            (r"\GPU Adapter Memory(*)\Dedicated Usage", self._dedicated),
            (r"\GPU Adapter Memory(*)\Shared Usage", self._shared),
        ):
            hr = self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(target))
            if hr != 0:
                raise OSError(f"PdhAddEnglishCounterW({path}) failed: 0x{hr & 0xFFFFFFFF:08x}")
        hr = self._pdh.PdhCollectQueryData(self._query)
        if hr != 0:
            raise OSError(f"PdhCollectQueryData failed: 0x{hr & 0xFFFFFFFF:08x}")

    def _read_counter(self, counter: ctypes.c_void_p) -> Optional[int]:
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        hr = self._pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None,
        )
        if (hr & 0xFFFFFFFF) != _PDH_MORE_DATA or size.value == 0:
            raise OSError(f"PdhGetFormattedCounterArrayW(size) failed: 0x{hr & 0xFFFFFFFF:08x}")
        buffer = ctypes.create_string_buffer(size.value)
        hr = self._pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), buffer,
        )
        if hr != 0:
            raise OSError(f"PdhGetFormattedCounterArrayW(data) failed: 0x{hr & 0xFFFFFFFF:08x}")
        items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        for i in range(count.value):
            name = (items[i].szName or "").lower()
            if self._needle in name:
                return max(0, int(round(items[i].doubleValue)))
        raise KeyError(f"PDH instance {self._needle!r} not found")

    def sample(self) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        if self.error or not self._query.value:
            return None, None, self.error
        try:
            hr = self._pdh.PdhCollectQueryData(self._query)
            if hr != 0:
                raise OSError(f"PdhCollectQueryData failed: 0x{hr & 0xFFFFFFFF:08x}")
            return self._read_counter(self._dedicated), self._read_counter(self._shared), None
        except Exception as exc:
            message = f"PDH 采样失败: {type(exc).__name__}: {exc}"
            if self.error is None:
                self.error = message
                log.warning("[MemoryMonitor] %s", message, exc_info=True)
            return None, None, message

    def close(self) -> None:
        if self._query.value:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except Exception:
                log.debug("PdhCloseQuery failed", exc_info=True)
            self._query = ctypes.c_void_p()


class SystemMetricsCollector:
    """Collect system-wide RAM/commit and NVIDIA adapter VRAM."""

    def __init__(self) -> None:
        self._kernel32 = None
        self._pdh: Optional[_PDHGPUMemory] = None
        self._gpu_info: Dict[str, Any] = {
            "gpu_name": None, "dedicated_total": None, "dedicated_budget": None,
            "shared_budget": None, "luid_low": None, "luid_high": None, "error": None,
        }
        self._static_error: Optional[str] = None
        if os.name == "nt":
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
            self._kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
            self._gpu_info = _read_dxgi_info()
            if self._gpu_info.get("luid_low") is not None:
                self._pdh = _PDHGPUMemory(
                    int(self._gpu_info["luid_low"]), int(self._gpu_info["luid_high"]),
                )
        else:
            self._static_error = "系统级 commit/共享显存仅支持 Windows"

    def collect(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ram_used": None, "ram_total": None,
            "commit_used": None, "commit_limit": None,
            "gpu_name": self._gpu_info.get("gpu_name"),
            "vram_dedicated_used": None,
            "vram_dedicated_total": self._gpu_info.get("dedicated_total"),
            "vram_dedicated_budget": self._gpu_info.get("dedicated_budget"),
            "vram_shared_used": None,
            "vram_shared_budget": self._gpu_info.get("shared_budget"),
            "error": self._static_error,
        }
        ram_error = None
        if self._kernel32 is not None:
            status = _MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if self._kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                data["ram_total"] = int(status.ullTotalPhys)
                data["ram_used"] = max(0, int(status.ullTotalPhys) - int(status.ullAvailPhys))
                data["commit_limit"] = int(status.ullTotalPageFile)
                data["commit_used"] = max(0, int(status.ullTotalPageFile) - int(status.ullAvailPageFile))
            else:
                ram_error = f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}"
        else:
            try:
                import psutil
                vm = psutil.virtual_memory()
                data["ram_total"] = int(vm.total)
                data["ram_used"] = int(vm.total - vm.available)
            except Exception as exc:
                ram_error = f"RAM 采集失败: {type(exc).__name__}: {exc}"

        gpu_error = self._gpu_info.get("error")
        if self._pdh is not None:
            dedicated, shared, pdh_error = self._pdh.sample()
            data["vram_dedicated_used"] = dedicated
            data["vram_shared_used"] = shared
            gpu_error = gpu_error or pdh_error
        data["error"] = ram_error or gpu_error
        return data

    def close(self) -> None:
        if self._pdh is not None:
            self._pdh.close()
            self._pdh = None
