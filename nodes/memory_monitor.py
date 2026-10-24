"""System-wide memory / VRAM monitor nodes (start + end).

Both nodes are pure Any passthroughs: they do not start/stop the sampler in
execute(), so their cache signatures stay stable and downstream caches are not
invalidated.  The frontend starts recording on execution_start and stops on
execution_success/error/interrupted via the routes below.  API callers can call
the same routes explicitly.
"""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web
from comfy_api.latest import io
from server import PromptServer

import folder_paths

from .memory_monitor_core import (
    DEFAULT_INTERVAL_S,
    DEFAULT_MAX_DURATION_MIN,
    MAX_LOGS,
    MemoryMonitor,
)

log = logging.getLogger("psl_comfyui_utilities.memory_monitor")

_LOG_DIR = folder_paths.get_user_directory() + "/psl_memory_monitor"
_MONITOR = MemoryMonitor(_LOG_DIR, max_logs=MAX_LOGS)


if getattr(PromptServer, "instance", None) is not None:

    @PromptServer.instance.routes.get("/psl/memory_monitor/runs")
    async def psl_memory_monitor_runs(request):
        try:
            limit = int(request.query.get("limit", str(MAX_LOGS)))
        except (TypeError, ValueError):
            limit = MAX_LOGS
        limit = max(1, min(limit, MAX_LOGS))
        return web.json_response({
            "runs": _MONITOR.list_runs(limit),
            "log_dir": str(_MONITOR.log_dir),
        })

    @PromptServer.instance.routes.get("/psl/memory_monitor/run")
    async def psl_memory_monitor_run(request):
        run_id = request.query.get("run_id", "")
        payload = _MONITOR.load_run(run_id)
        if payload is None:
            return web.json_response({"error": "未找到该次记录"}, status=404)
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/psl/memory_monitor/start")
    async def psl_memory_monitor_start(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        interval = body.get("interval_s", DEFAULT_INTERVAL_S) if isinstance(body, dict) else DEFAULT_INTERVAL_S
        max_duration_min = body.get("max_duration_min", DEFAULT_MAX_DURATION_MIN) if isinstance(body, dict) else DEFAULT_MAX_DURATION_MIN
        run_id = await asyncio.to_thread(_MONITOR.start, interval, max_duration_min)
        if not run_id:
            return web.json_response(
                {"error": _MONITOR.init_error or "监控启动失败"}, status=500,
            )
        return web.json_response({"run_id": run_id})

    @PromptServer.instance.routes.post("/psl/memory_monitor/stop")
    async def psl_memory_monitor_stop(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = str(body.get("reason", "end_node")) if isinstance(body, dict) else "end_node"
        payload = await asyncio.to_thread(_MONITOR.stop, reason)
        if payload is None:
            return web.json_response({"run_id": "", "reason": reason})
        return web.json_response({
            "run_id": payload.get("run_id", ""),
            "reason": payload.get("end", {}).get("reason", reason),
            "samples": len(payload.get("samples", [])),
        })


class MemoryMonitorStart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MemoryMonitorStart",
            display_name="Memory Monitor Start",
            category="utils/monitor",
            description=(
                "内存/显存监控的参数面板与 Any 直通。采样间隔/最长记录时间在节点的"
                "前端面板里设置（存在 node.properties，不进 prompt），因此改参数不会"
                "改变缓存签名、不会让下游缓存失效。前端在 prompt 执行开始时读取面板值"
                "并调用 /psl/memory_monitor/start；执行结束后停止。API 调用需要自行"
                "调用 start/stop 路由。"
            ),
            inputs=[
                io.AnyType.Input(
                    "any", optional=True,
                    tooltip="任意类型直通。把它串在工作流开头；Start 会原样输出。",
                ),
            ],
            outputs=[io.AnyType.Output("any", display_name="any")],
        )

    @classmethod
    def execute(cls, any=None):
        return io.NodeOutput(any)


class MemoryMonitorEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MemoryMonitorEnd",
            display_name="Memory Monitor End",
            category="utils/monitor",
            is_output_node=True,
            description=(
                "内存/显存监控的 Any 直通与显示节点。本身不停止采样；前端在 prompt"
                "结束/报错/中断时调用 /psl/memory_monitor/stop 并刷新最近 3 次曲线。"
                "节点即使被缓存跳过，显示仍由前端事件更新。"
            ),
            inputs=[
                io.AnyType.Input(
                    "any", optional=True,
                    tooltip="任意类型直通。通常接在监控链路的末尾。",
                ),
            ],
            outputs=[io.AnyType.Output("any", display_name="any")],
        )

    @classmethod
    def execute(cls, any=None):
        return io.NodeOutput(any)
