"""Persistent Output — 跨运行保留输入的桥接/回放节点（3 组「图像+音频」槽位）。

设计给「对比多轮视频结果」用：每组槽位是图像批次+音频的一对，规则相同且相互独立：

- 本轮该槽有输入：存储替换为新输入，输出 = 输入（纯桥接）
- 本轮该槽无输入（上游断开或上游给了 None）：输出 = 该槽上一次保留的内容
- 该槽从未保留过：输出 None（下游收到 None，等价于没连线；下游需容忍 None）

典型对比流：轮 1 接上游跑（槽 1 保留+桥接）→ 改参数跑轮 2（槽 2）→ 再跑轮 3（槽 3）
→ 断开上游：槽 1/槽 2 的输出分别接 A/B Compare 两耳，与实时结果并排对比。

内存策略（读码确认后的取舍）：
- 只存内存（模块级 dict），不写盘；ComfyUI 进程退出即全部释放。
- 键是节点 unique_id（IO.Hidden.unique_id，与前端 node.id 一致），每个节点
  实例一个独立槽位组，互不干扰。
- 存引用不 clone：ComfyUI 惯例输入张量不可变（节点都返回新张量），clone 会让
  内存翻倍。若某个下游节点不守惯例原地改写，保留的快照会被一起改掉——这是
  换取内存的已知取舍。
- 替换即释放：存新值时旧张量引用被覆盖，无其他持有者就立刻回收。
- 手动释放：前端按钮走 /psl/persist/release，按槽释放或全清；释放必须干净
  （三槽全空则整个键删掉）；节点从画布删除时前端自动调一次全清。
"""

import logging

from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

SLOT_COUNT = 3

# unique_id -> [ {image, audio} × SLOT_COUNT ]
# 只存内存：不序列化、不落盘，进程退出随进程消失。
_STORE: dict[str, list[dict]] = {}


def _slots(uid: str) -> list[dict]:
    return _STORE.setdefault(uid, [{"image": None, "audio": None} for _ in range(SLOT_COUNT)])


def store_input(uid: str, slot: int, image=None, audio=None) -> None:
    """本轮该槽有输入：替换保留内容（旧引用被覆盖，无其他持有者即回收）。"""
    if uid is None or not (0 <= slot < SLOT_COUNT):
        return
    s = _slots(uid)[slot]
    if image is not None:
        s["image"] = image
    if audio is not None:
        s["audio"] = audio


def release(uid: str, target: str = "all") -> bool:
    """手动释放。target: "slot1"/"slot2"/"slot3" / "all"。返回是否确有内容被释放。

    删键/清槽即释放：张量引用被丢弃，Python 引用计数负责回收（若预览缓存等处
    还持有引用，那部分内存随持有者消失，本节点不阻塞它）。
    """
    if uid is None or uid not in _STORE:
        return False
    if target == "all":
        del _STORE[uid]
        return True
    idx = {"slot1": 0, "slot2": 1, "slot3": 2}.get(target)
    if idx is None:
        return False
    slots = _STORE.get(uid)
    if slots is None:
        return False
    had = slots[idx]["image"] is not None or slots[idx]["audio"] is not None
    slots[idx] = {"image": None, "audio": None}
    # 三槽全空就把整个键删掉，不留空壳
    if all(s["image"] is None and s["audio"] is None for s in slots):
        _STORE.pop(uid, None)
    return had


def _describe_channel(kind: str, value) -> str:
    if value is None:
        return "未保留"
    if kind == "image":
        b, h, w = value.shape[0], value.shape[1], value.shape[2]
        return f"{b} 帧 {w}×{h}"
    wf = value.get("waveform")
    sr = value.get("sample_rate")
    if wf is None or not sr:
        return "已保留"
    ch = wf.shape[1]
    dur = wf.shape[-1] / float(sr)
    return f"{sr} Hz · {dur:.1f}s · {ch} 声道"


def describe(uid) -> list[str]:
    """给前端按钮 tooltip/状态用的人类可读描述，每槽一条。"""
    if uid is None or uid not in _STORE:
        return ["未保留"] * SLOT_COUNT
    return [
        "、".join(filter(
            lambda t: t != "未保留",
            (_describe_channel("image", s["image"]), _describe_channel("audio", s["audio"])),
        )) or "未保留"
        for s in _STORE[uid]
    ]


class PersistentOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PersistentOutput",
            search_aliases=["persist", "保留", "记忆", "回放", "桥接", "跨运行", "对比", "replay"],
            display_name="Persistent Output",
            category="utils/compare",
            description=(
                "跨运行保留输入（3 组槽位，每组=图像批次+音频）：某槽本轮有输入则桥接"
                "输出并替换保留；断开上游后该槽输出上一次保留的内容；从未保留则输出"
                " None。给对比多轮视频结果用：每轮把上游接到一个空槽跑（保留+桥接），"
                "然后断开上游——各槽吐出各自保留的内容，两组输出接 A/B Compare 两耳"
                "即可同时对比。\n"
                "保留内容只存内存不写盘，ComfyUI 关闭即清空；新输入替换旧保留；按钮"
                "按槽释放，删除节点全清。下游节点需要容忍 None。槽位标注文本纯展示，"
                "不参与运行。"
            ),
            # 必须是输出节点：断开上游后节点仍会执行（输出节点不进缓存跳过），
            # 回放路径才真的跑得起来；也让「运行到此节点」可用。
            is_output_node=True,
            inputs=[
                # 交错排列：图像1,音频1,图像2,音频2,… 与每组语义对应，视觉上一组紧挨一组
                *[
                    inp
                    for i in range(1, SLOT_COUNT + 1)
                    for inp in (
                        io.Image.Input(
                            f"image_{i}",
                            display_name=f"图像 {i}",
                            optional=True,
                            tooltip=f"槽 {i} 的图像批次（视频帧）。有输入则替换保留并原样输出；断开后输出该槽保留内容。",
                        ),
                        io.Audio.Input(
                            f"audio_{i}",
                            display_name=f"音频 {i}",
                            optional=True,
                            tooltip=f"槽 {i} 的音频。有输入则替换保留并原样输出；断开后输出该槽保留内容。",
                        ),
                    )
                ],
                io.String.Input(
                    "labels_json",
                    default="[]",
                    optional=True,
                    tooltip=(
                        "三个槽位的用户标注文本（JSON 数组），由前端维护、随工作流文件"
                        "持久化；前端会把它标记为不参与 prompt 序列化，后端正常收不到，"
                        "仅作兼容保留。"
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                *[
                    out
                    for i in range(1, SLOT_COUNT + 1)
                    for out in (
                        io.Image.Output(f"image_{i}", display_name=f"图像 {i}"),
                        io.Audio.Output(f"audio_{i}", display_name=f"音频 {i}"),
                    )
                ]
            ],
        )

    @classmethod
    def execute(cls, image_1=None, audio_1=None,
                image_2=None, audio_2=None,
                image_3=None, audio_3=None, labels_json="[]"):
        uid = cls.hidden.unique_id
        if uid is None:
            raise ValueError(
                "PersistentOutput 拿不到节点 unique_id（仅在 ComfyUI 运行时可用）；"
                "请通过 ComfyUI 正常排队运行。"
            )
        del labels_json   # UI-only，后端不消费；形参仅防前端脚本未生效时报多余 kwarg

        inputs = [(image_1, audio_1), (image_2, audio_2), (image_3, audio_3)]
        outs = []
        for i, (img, aud) in enumerate(inputs):
            store_input(uid, i, image=img, audio=aud)
            slot = _slots(uid)[i]
            # 桥接 + 回放：有输入出输入，没输入出保留，都没有出 None
            outs.extend([
                img if img is not None else slot["image"],
                aud if aud is not None else slot["audio"],
            ])
        return io.NodeOutput(*outs)


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
        return web.json_response({"slots": describe(uid)})

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
        if target not in ("slot1", "slot2", "slot3", "all"):
            return web.json_response({"error": f"未知 target: {target!r}"}, status=400)
        had = release(uid, target)
        logging.info("[PersistentOutput] 释放 node=%s target=%s 有内容=%s", uid, target, had)
        return web.json_response({"released": had, "slots": describe(uid)})
