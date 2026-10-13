"""Persistent Output — 跨运行保留输入的桥接/回放节点（图像批次 + 音频）。

设计给「对比两轮视频结果」用：有输入时原样桥接并顺手保留；断开上游后，
下一次运行把上一轮保留的内容原样吐出来——于是：

  轮 1：接上游运行 → 保留视频帧 + 音频，同时正常喂给下游
  轮 2：改参数再跑 → 保留内容被新输入替换
  对比：断开上游，把本节点的保留输出接 Audio A/B Compare 左声道、
        本轮的实时输出接右声道（或接两个预览并排看），再运行

语义（图像与音频两条通道相互独立，规则相同）：
- 本轮有输入：存储替换为新输入，输出 = 输入（纯桥接）
- 本轮无输入（上游断开或上游给了 None）：输出 = 上一次保留的内容
- 从未保留过：输出 None（下游收到 None，等价于没连线；下游需容忍 None）

内存策略（读码确认后的取舍）：
- 只存内存（模块级 dict），不写盘；ComfyUI 进程退出即全部释放。
- 键是节点 unique_id（IO.Hidden.unique_id，与前端 node.id 一致），每个节点
  实例一个独立槽位，互不干扰。
- 存引用不 clone：ComfyUI 惯例输入张量不可变（节点都返回新张量），clone 会让
  内存翻倍。若某个下游节点不守惯例原地改写，保留的快照会被一起改掉——这是
  换取内存的已知取舍。
- 替换即释放：存新值时旧张量引用被覆盖，无其他持有者就立刻回收。
- 手动释放：前端按钮走 /psl/persist/release，把键整个删掉，释放必须干净；
  节点从画布删除时前端也会自动调一次（onRemoved）。
"""

import logging

import folder_paths
from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

# unique_id -> {"image": IMAGE 张量 | None, "audio": AUDIO dict | None}
# 只存内存：不序列化、不落盘，进程退出随进程消失。
_STORE: dict[str, dict] = {}

_IMAGE = "image"
_AUDIO = "audio"


def _slot(uid: str) -> dict:
    return _STORE.setdefault(uid, {_IMAGE: None, _AUDIO: None})


def store_input(uid: str, image=None, audio=None) -> None:
    """本轮有输入：替换保留内容（旧引用被覆盖，无其他持有者即回收）。"""
    if uid is None:
        return
    slot = _slot(uid)
    if image is not None:
        slot[_IMAGE] = image
    if audio is not None:
        slot[_AUDIO] = audio


def release(uid: str, target: str = "all") -> bool:
    """手动释放。target: image / audio / all。返回是否确有内容被释放。

    删键即释放：dict 与槽位里的张量引用一并丢弃，Python 引用计数负责回收
    （若预览缓存等处还持有引用，那部分内存随持有者消失，本节点不阻塞它）。
    """
    if uid is None or uid not in _STORE:
        return False
    if target == "all":
        del _STORE[uid]
        return True
    slot = _STORE.get(uid)
    if slot is None:
        return False
    had = slot.get(target) is not None
    slot[target] = None
    # 两条通道都空了就把整个键删掉，不留空壳
    if slot.get(_IMAGE) is None and slot.get(_AUDIO) is None:
        _STORE.pop(uid, None)
    return had


def describe(uid) -> dict:
    """给前端状态行用的人类可读描述；本节点从不写入 filename。"""
    slot = _STORE.get(str(uid)) if uid is not None else None
    out = {_IMAGE: "未保留", _AUDIO: "未保留"}
    if not slot:
        return out
    img = slot.get(_IMAGE)
    if img is not None:
        b, h, w = img.shape[0], img.shape[1], img.shape[2]
        out[_IMAGE] = f"{b} 帧 {w}×{h}"
    aud = slot.get(_AUDIO)
    if aud is not None:
        wf = aud.get("waveform")
        sr = aud.get("sample_rate")
        if wf is not None and sr:
            ch = wf.shape[1]
            dur = wf.shape[-1] / float(sr)
            out[_AUDIO] = f"{sr} Hz · {dur:.1f}s · {ch} 声道"
        else:
            out[_AUDIO] = "已保留"
    return out


class PersistentOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PersistentOutput",
            search_aliases=["persist", "保留", "记忆", "回放", "桥接", "跨运行", "对比", "replay"],
            display_name="Persistent Output",
            category="utils/compare",
            description=(
                "跨运行保留输入：有输入时原样桥接输出并顺手保留；断开上游后，下一次"
                "运行把上一轮保留的内容原样输出；从未保留过则输出 None。图像与音频"
                "两条通道相互独立。给对比两轮视频结果用：轮 1 接上游跑（保留+桥接），"
                "改参数跑轮 2（自动替换保留），然后断开上游——本节点吐出上一轮内容，"
                "接 A/B Compare 或双预览与新一轮结果并排对比。\n"
                "保留内容只存在内存里，不写盘，ComfyUI 关闭即清空；新输入会替换旧"
                "保留；按钮或删除节点可手动释放。下游节点需要容忍 None（从未保留且"
                "无输入时本节点输出 None）。"
            ),
            # 必须是输出节点：断开上游后节点仍会执行（输出节点不进缓存跳过），
            # 回放路径才真的跑得起来；也让「运行到此节点」可用。
            is_output_node=True,
            inputs=[
                io.Image.Input(
                    "image_input",
                    display_name="图像批次",
                    optional=True,
                    tooltip=(
                        "要保留/桥接的图像批次（视频帧）。本轮有输入则替换保留并原样"
                        "输出；断开后输出上一次保留的内容。"
                    ),
                ),
                io.Audio.Input(
                    "audio_input",
                    display_name="音频",
                    optional=True,
                    tooltip=(
                        "要保留/桥接的音频。本轮有输入则替换保留并原样输出；断开后"
                        "输出上一次保留的内容。"
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                io.Image.Output("image", display_name="图像"),
                io.Audio.Output("audio", display_name="音频"),
            ],
        )

    @classmethod
    def execute(cls, image_input=None, audio_input=None):
        uid = cls.hidden.unique_id
        if uid is None:
            raise ValueError(
                "PersistentOutput 拿不到节点 unique_id（仅在 ComfyUI 运行时可用）；"
                "请通过 ComfyUI 正常排队运行。"
            )

        # 桥接 + 保留：有输入就替换存储（引用直存，见模块 docstring 的取舍说明）
        store_input(uid, image=image_input, audio=audio_input)
        slot = _slot(uid)

        # 无输入时回放保留内容；从未保留则 None（下游收到 None = 等价没连线）
        image_out = image_input if image_input is not None else slot.get(_IMAGE)
        audio_out = audio_input if audio_input is not None else slot.get(_AUDIO)
        return io.NodeOutput(image_out, audio_out)


# ---------------------------------------------------------------------------
# 状态查询 / 手动释放路由（前端按钮与节点删除清理用）
# ---------------------------------------------------------------------------
def _parse_uid(raw) -> str | None:
    """节点 id 统一转字符串；非法输入返回 None。"""
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


if getattr(PromptServer, "instance", None) is not None:

    @PromptServer.instance.routes.get("/psl/persist/status")
    async def persist_status(request):
        uid = _parse_uid(request.query.get("node_id"))
        if uid is None:
            return web.json_response({"error": "缺少 node_id"}, status=400)
        return web.json_response(describe(uid))

    @PromptServer.instance.routes.post("/psl/persist/release")
    async def persist_release(request):
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"请求体不是合法 JSON: {e}"}, status=400)
        uid = _parse_uid(body.get("node_id") if isinstance(body, dict) else None)
        target = body.get("target", "all") if isinstance(body, dict) else "all"
        if uid is None:
            return web.json_response({"error": "缺少 node_id"}, status=400)
        if target not in (_IMAGE, _AUDIO, "all"):
            return web.json_response({"error": f"未知 target: {target!r}"}, status=400)
        had = release(uid, target)
        logging.info("[PersistentOutput] 释放 node=%s target=%s 有内容=%s", uid, target, had)
        return web.json_response({"released": had, "held": describe(uid)})
