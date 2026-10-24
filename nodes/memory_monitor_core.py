"""Sampling thread, JSONL run log and history reader for the memory monitor.

No ComfyUI imports here; the node/routes layer lives in memory_monitor.py.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .system_memory import SystemMetricsCollector
except ImportError:  # standalone test: nodes/ directory is on sys.path
    from system_memory import SystemMetricsCollector

log = logging.getLogger("psl_comfyui_utilities.memory_monitor")

MIN_INTERVAL_S = 0.2
MAX_INTERVAL_S = 60.0
DEFAULT_INTERVAL_S = 1.0
DEFAULT_MAX_DURATION_MIN = 120
MAX_LOGS = 3
_RUN_ID_RE = re.compile(r"^[0-9A-Za-z_]+$")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_line(file_obj, payload: Dict[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    file_obj.flush()


@dataclass
class _Run:
    run_id: str
    path: Path
    interval_s: float
    max_duration_min: int
    max_duration_s: float
    started_monotonic: float
    started_epoch_ms: int
    started_at: str
    meta: Dict[str, Any] = field(default_factory=dict)
    end: Optional[Dict[str, Any]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    error: Optional[str] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    file_obj: Any = None
    thread: Optional[threading.Thread] = None

    def payload(self) -> Dict[str, Any]:
        with self.lock:
            meta = dict(self.meta)
            end = dict(self.end) if self.end else {
                "reason": self.reason or "incomplete",
                "samples": len(self.samples),
                "warnings": list(self.warnings),
            }
            meta["file"] = self.path.name
            return {
                "run_id": self.run_id,
                "meta": meta,
                "samples": list(self.samples),
                "end": end,
            }


class MemoryMonitor:
    def __init__(self, log_dir: Path | str, max_logs: int = MAX_LOGS):
        self.log_dir = Path(log_dir)
        self.max_logs = max(1, int(max_logs))
        self._lock = threading.RLock()
        self._run: Optional[_Run] = None
        self._last: Optional[_Run] = None
        self._init_error: Optional[str] = None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._init_error = f"日志目录不可用: {type(exc).__name__}: {exc}"
            log.error("[MemoryMonitor] %s", self._init_error, exc_info=True)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    # ------------------------------------------------------------------
    #  lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_duration_min: int = DEFAULT_MAX_DURATION_MIN,
        *,
        max_duration_s: Optional[float] = None,  # test hook
    ) -> str:
        if self._init_error:
            with self._lock:
                self._last = None
                self._run = None
            log.error("[MemoryMonitor] 无法开始记录: %s", self._init_error)
            return ""
        try:
            interval = float(interval_s)
        except (TypeError, ValueError):
            interval = DEFAULT_INTERVAL_S
        interval = min(max(interval, MIN_INTERVAL_S), MAX_INTERVAL_S)
        try:
            duration_min = int(max_duration_min)
        except (TypeError, ValueError):
            duration_min = DEFAULT_MAX_DURATION_MIN
        duration_min = min(max(duration_min, 1), 1440)
        duration_s = float(max_duration_s) if max_duration_s else duration_min * 60.0

        previous: Optional[_Run] = None
        with self._lock:
            if self._run is not None:
                if not self._run.finished_event.is_set():
                    previous = self._run
                    previous.reason = previous.reason or "superseded"
                    previous.stop_event.set()
                else:
                    self._last = self._run
                    self._run = None
        if previous is not None and previous.thread is not None:
            previous.thread.join(timeout=6)
            if previous.thread.is_alive():
                log.error("[MemoryMonitor] 上一轮采样线程未在 6 秒内退出: %s", previous.run_id)

        self._prune(self.max_logs - 1)
        now = time.time()
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + "_" + secrets.token_hex(3)
        state = _Run(
            run_id=run_id,
            path=self.log_dir / f"run_{run_id}.jsonl",
            interval_s=interval,
            max_duration_min=duration_min,
            max_duration_s=duration_s,
            started_monotonic=time.monotonic(),
            started_epoch_ms=int(now * 1000),
            started_at=_now_iso(),
        )
        state.meta = {
            "type": "meta",
            "schema": 1,
            "run_id": run_id,
            "started_at": state.started_at,
            "start_epoch_ms": state.started_epoch_ms,
            "interval_s": interval,
            "max_duration_min": duration_min,
            "max_duration_s": duration_s,
            "gpu_name": None,
        }
        with self._lock:
            self._run = state
        thread = threading.Thread(
            target=self._sampler_main, args=(state,),
            name=f"psl-memory-monitor-{run_id}", daemon=True,
        )
        state.thread = thread
        thread.start()
        # Wait briefly so the GPU name/meta line is usually ready before the
        # workflow advances; a slow collector must not block the prompt.
        state.ready_event.wait(timeout=0.75)
        log.debug("[MemoryMonitor] started %s interval=%.2fs max=%.0fs", run_id, interval, duration_s)
        return run_id

    def stop(self, reason: str = "end_node") -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._run
            if state is None:
                return self._last.payload() if self._last is not None else None
            if not state.finished_event.is_set():
                if state.reason is None:
                    state.reason = reason
                state.stop_event.set()
        if state.thread is not None:
            state.thread.join(timeout=8)
            if state.thread.is_alive():
                log.error("[MemoryMonitor] 采样线程未在 8 秒内退出: %s", state.run_id)
        with self._lock:
            self._last = state
            self._run = None
        self._prune(self.max_logs)
        payload = state.payload()
        log.debug("[MemoryMonitor] stopped %s reason=%s samples=%d",
                  state.run_id, payload.get("end", {}).get("reason"), len(payload.get("samples", [])))
        return payload

    def current_run_id(self) -> Optional[str]:
        with self._lock:
            return self._run.run_id if self._run is not None else None

    # ------------------------------------------------------------------
    #  sampler
    # ------------------------------------------------------------------

    def _append_sample_locked(self, state: _Run, sample: Dict[str, Any], elapsed: float) -> None:
        row = dict(sample)
        row["t"] = round(max(0.0, elapsed), 3)
        row["ts"] = int(time.time() * 1000)
        sample_error = row.get("error")
        if sample_error and sample_error not in state.warnings:
            state.warnings.append(str(sample_error))
            log.warning("[MemoryMonitor] 采样指标异常: %s", sample_error)
        state.samples.append(row)
        if state.file_obj is not None:
            try:
                _write_json_line(state.file_obj, {"type": "sample", **row})
            except Exception as exc:
                message = f"日志写入失败: {type(exc).__name__}: {exc}"
                if message not in state.warnings:
                    state.warnings.append(message)
                    log.error("[MemoryMonitor] %s", message, exc_info=True)

    def _sampler_main(self, state: _Run) -> None:
        collector: Optional[SystemMetricsCollector] = None
        try:
            collector = SystemMetricsCollector()
            first = collector.collect()
            with state.lock:
                state.meta["gpu_name"] = first.get("gpu_name")
                try:
                    state.file_obj = state.path.open("w", encoding="utf-8", newline="\n", buffering=1)
                    _write_json_line(state.file_obj, state.meta)
                except Exception as exc:
                    message = f"日志文件打开失败: {type(exc).__name__}: {exc}"
                    state.warnings.append(message)
                    log.error("[MemoryMonitor] %s", message, exc_info=True)
                    state.file_obj = None
                self._append_sample_locked(state, first, 0.0)
            state.ready_event.set()

            next_deadline = state.started_monotonic + state.interval_s
            while not state.stop_event.is_set():
                now = time.monotonic()
                elapsed = now - state.started_monotonic
                if elapsed >= state.max_duration_s:
                    state.reason = state.reason or "timeout"
                    break
                wait_for = min(next_deadline - now, state.max_duration_s - elapsed)
                if wait_for > 0 and state.stop_event.wait(wait_for):
                    break
                if state.stop_event.is_set():
                    break
                elapsed = time.monotonic() - state.started_monotonic
                if elapsed >= state.max_duration_s:
                    state.reason = state.reason or "timeout"
                    break
                try:
                    sample = collector.collect()
                except Exception as exc:
                    message = f"采样失败: {type(exc).__name__}: {exc}"
                    with state.lock:
                        if message not in state.warnings:
                            state.warnings.append(message)
                            log.error("[MemoryMonitor] %s", message, exc_info=True)
                    sample = None
                if sample is not None:
                    with state.lock:
                        self._append_sample_locked(state, sample, elapsed)
                next_deadline += state.interval_s
        except Exception as exc:
            message = f"采样线程异常: {type(exc).__name__}: {exc}"
            with state.lock:
                state.error = message
                state.warnings.append(message)
            log.error("[MemoryMonitor] %s", message, exc_info=True)
        finally:
            elapsed = time.monotonic() - state.started_monotonic
            if collector is not None:
                try:
                    final = collector.collect()
                    with state.lock:
                        last_t = state.samples[-1]["t"] if state.samples else -1.0
                        if final is not None and elapsed - last_t >= 0.01:
                            self._append_sample_locked(state, final, elapsed)
                except Exception as exc:
                    with state.lock:
                        state.warnings.append(f"收尾采样失败: {type(exc).__name__}: {exc}")
                finally:
                    collector.close()
            with state.lock:
                if state.reason is None:
                    state.reason = "error" if state.error else "ended"
                state.end = {
                    "type": "end",
                    "ended_at": _now_iso(),
                    "ended_epoch_ms": int(time.time() * 1000),
                    "duration_s": round(elapsed, 3),
                    "samples": len(state.samples),
                    "reason": state.reason,
                    "warnings": list(state.warnings),
                }
                if state.file_obj is not None:
                    try:
                        _write_json_line(state.file_obj, state.end)
                    except Exception as exc:
                        log.error("[MemoryMonitor] 结束行写入失败: %s", exc, exc_info=True)
                    try:
                        state.file_obj.close()
                    except Exception:
                        log.debug("log close failed", exc_info=True)
                    state.file_obj = None
                state.finished_event.set()
                state.ready_event.set()

    # ------------------------------------------------------------------
    #  logs / history
    # ------------------------------------------------------------------

    def _prune(self, keep: int) -> None:
        keep = max(1, int(keep))
        try:
            files = sorted(
                (p for p in self.log_dir.glob("run_*.jsonl") if p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            for path in files[keep:]:
                try:
                    path.unlink()
                    log.info("[MemoryMonitor] 删除旧日志 %s", path.name)
                except Exception as exc:
                    log.warning("[MemoryMonitor] 删除旧日志失败 %s: %s", path, exc)
        except Exception as exc:
            log.warning("[MemoryMonitor] 清理旧日志失败: %s", exc, exc_info=True)

    def _parse_log(self, path: Path) -> Optional[Dict[str, Any]]:
        meta = None
        samples: List[Dict[str, Any]] = []
        end = None
        warnings: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception as exc:
                        warnings.append(f"第 {line_no} 行解析失败: {exc}")
                        continue
                    kind = row.get("type")
                    if kind == "meta" and meta is None:
                        meta = row
                    elif kind == "sample":
                        row.pop("type", None)
                        samples.append(row)
                    elif kind == "end":
                        end = row
        except Exception as exc:
            log.warning("[MemoryMonitor] 读取日志失败 %s: %s", path, exc, exc_info=True)
            return None
        if meta is None:
            return None
        if end is None:
            end = {
                "reason": "incomplete",
                "samples": len(samples),
                "warnings": warnings,
            }
        else:
            end = dict(end)
            if warnings:
                end.setdefault("warnings", [])
                end["warnings"] = list(end["warnings"]) + warnings
        meta = dict(meta)
        meta["file"] = path.name
        return {
            "run_id": meta.get("run_id") or path.stem.removeprefix("run_"),
            "meta": meta,
            "samples": samples,
            "end": end,
        }

    def list_runs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        limit = self.max_logs if limit is None else max(1, int(limit))
        runs: List[Dict[str, Any]] = []
        for path in self.log_dir.glob("run_*.jsonl"):
            parsed = self._parse_log(path)
            if parsed is None:
                continue
            meta = parsed["meta"]
            end = parsed["end"]
            runs.append({
                "run_id": parsed["run_id"],
                "started_at": meta.get("started_at"),
                "start_epoch_ms": meta.get("start_epoch_ms"),
                "ended_epoch_ms": end.get("ended_epoch_ms"),
                "duration_s": end.get("duration_s"),
                "interval_s": meta.get("interval_s"),
                "gpu_name": meta.get("gpu_name"),
                "samples": end.get("samples", len(parsed["samples"])),
                "reason": end.get("reason", "incomplete"),
                "file": path.name,
            })
        runs.sort(key=lambda r: (r.get("start_epoch_ms") or 0), reverse=True)
        return runs[:limit]

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not run_id or not _RUN_ID_RE.fullmatch(run_id):
            return None
        path = (self.log_dir / f"run_{run_id}.jsonl").resolve()
        try:
            path.relative_to(self.log_dir.resolve())
        except ValueError:
            return None
        if not path.is_file():
            return None
        return self._parse_log(path)
