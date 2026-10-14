"""Persistent Output — 跨运行保留输入的桥接/回放节点（3 组「图像+音频」槽位）。

设计给「对比多轮视频结果」用：每组槽位是图像批次+音频的一对，规则相同且相互独立：

- 本轮该槽有输入：存储替换为新输入，输出 = 输入（纯桥接）
- 本轮该槽无输入（上游断开或上游给了 None）：输出 = 该槽上一次保留的内容
- 该槽从未保留过：输出 None（下游收到 None，等价于没连线；下游需容忍 None）

典型对比流：轮 1 接上游跑（槽 1 保留+桥接）→ 改参数跑轮 2（槽 2）→ 再跑轮 3（槽 3）
→ 断开上游：槽 1/槽 2 的输出分别接 A/B Compare 两耳，与实时结果并排对比。

内存策略（用户实测踩坑后定稿——存储必须用**深拷贝**而不是引用）：
- ComfyUI 会把本节点的输出也写进全局运行缓存（execution.py:616-618，CacheEntry
  跨运行长期存活，缓存命中时 execute 根本不重跑——execution.py:447）。若存引用，
  _STORE 与缓存条目持有同一批张量对象，释放 _STORE 后缓存条目仍攥着它们，
  内存纹丝不动（"霸占"）。
- 所以：存入时 clone 一份节点独占的副本；释放时除清 _STORE 外，还把**本节点自己**
  的输出缓存条目一并摘除。上游节点自己的缓存完全不碰——那份原件由 ComfyUI 自己的
  缓存策略（或 DeepVRAMCleanup）管辖。
- 输出恒等于存储副本：桥接时也不直出上游原件。否则本节点的输出缓存条目会 pin 住
  上游对象——内核 RAM pressure 缓存按条目大小记账驱逐（caching.py ram_release），
  上游条目被驱逐时原件仍被我们的条目攥着，实际释放不掉，内核以为已释放、压力不解除
  就加码驱逐，把上游节点（VHS）自己的缓存也反复清掉重算——这就是"独立缓存与内核
  缓存严重冲突"的根源（实测踩坑：VHS 反复重解码、释放后内存不动直到压满）。
- 摘缓存有两个前提（实测踩坑后补齐）：① schema 必须声明 io.Hidden.execution_list，
  否则内核不注入执行列表，句柄永远是 None，摘除沦为空转（HiddenHolder 对未声明的
  hidden 一律返回 None）；② gen widget 让签名每轮必变 ⇒ 每轮产生一个**新 data key**
  的缓存条目，而内核默认 RAM pressure 缓存只在系统内存告急时才回收旧条目——所以
  必须自己记账历轮 key：execute 时清掉旧轮条目，release 时全数摘除。
- 代价：桥接期间内容在内存里存两份（本节点副本 + 上游缓存原件），这是隔离
  的必要成本；换来的释放是真正的干净。
- 其余不变：只存内存不落盘，进程退出即释放；键是节点 unique_id；三槽全空
  删整个键；节点从画布删除时前端自动全清。
"""

import logging

import torch
from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

SLOT_COUNT = 3

# unique_id -> [ {image, audio} × SLOT_COUNT ]
# 只存内存：不序列化、不落盘，进程退出随进程消失。
_STORE: dict[str, list[dict]] = {}

# 存储代数：每次 release 递增。进入缓存签名（见 execute 的 gen widget），
# 让「release 之后的第一轮」缓存必然 miss 重跑，前端拿到新鲜 ui 而不是
# 上一轮的旧快照（缓存命中重放旧 ui 是实测踩过的坑）。
_GENERATION: dict[str, int] = {}


def _bump_generation(uid: str) -> None:
    _GENERATION[uid] = _GENERATION.get(uid, 0) + 1

# unique_id -> 本节点最近一次执行时的 outputs 缓存对象（全局 CacheSet 的成员，
# 跨运行长期存活）。释放路由没有 hidden 上下文，只能靠这里存的句柄去摘
# 本节点自己的缓存条目。句柄来自 hidden.execution_list——schema 里**必须声明**
# io.Hidden.execution_list，否则内核不注入，这里永远是 None（踩过的坑）。
_CACHE_HANDLES: dict[str, object] = {}

# unique_id -> 本节点历轮写入 outputs 缓存的全部 data key。
# gen widget 让签名每轮必变 ⇒ 每轮都会产生一个**新 key** 的缓存条目；内核默认
# RAM pressure 缓存只在系统内存告急时才回收旧条目，平时任由旧条目攥着上游原图/
# 回放副本——这是"反复导入后内存持续上涨"的根源。这里记账历轮 key：
# execute 时清掉旧轮条目，release 时全数摘除。
_CACHE_KEYS: dict[str, set] = {}


def _slots(uid: str) -> list[dict]:
    return _STORE.setdefault(uid, [{"image": None, "audio": None} for _ in range(SLOT_COUNT)])


def _clone_value(kind: str, value):
    """存入前深拷贝：副本归本节点独占，释放才能真降内存（见模块 docstring）。"""
    if value is None:
        return None
    if kind == "image":
        return value.clone() if isinstance(value, torch.Tensor) else value
    out = dict(value)
    wf = out.get("waveform")
    if isinstance(wf, torch.Tensor):
        out["waveform"] = wf.clone()
    return out


def _tensor_bytes(t) -> int:
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def _mem_brief(kind: str, original, stored) -> str:
    """存入诊断：确认副本真的新建了独立存储。device 决定内存落在 RAM 还是显存，
    data_ptr 相同则说明没拷贝（与输入共享存储）——内存不涨的判决性证据。"""
    if kind == "image":
        if isinstance(original, torch.Tensor) and isinstance(stored, torch.Tensor):
            shared = original.data_ptr() == stored.data_ptr()
            return (f"tensor dev={stored.device} dtype={stored.dtype} "
                    f"shape={tuple(stored.shape)} bytes={_tensor_bytes(stored):,}"
                    + (" ⚠与输入共享存储，未真正拷贝!" if shared else ""))
        return f"⚠非张量 original={type(original).__name__} stored={type(stored).__name__}"
    wf = stored.get("waveform") if isinstance(stored, dict) else None
    if isinstance(wf, torch.Tensor):
        return f"audio dict waveform dev={wf.device} shape={tuple(wf.shape)} bytes={_tensor_bytes(wf):,}"
    return f"⚠audio 非 dict/无波形: {type(stored).__name__}"


def _slot_bytes(s: dict) -> int:
    return _tensor_bytes(s.get("image")) + (
        _tensor_bytes(s["audio"].get("waveform")) if isinstance(s.get("audio"), dict) else 0)


def store_input(uid: str, slot: int, image=None, audio=None) -> None:
    """本轮该槽有输入：把**深拷贝副本**存进保留区（原件留给上游缓存与桥接输出）。"""
    if uid is None or not (0 <= slot < SLOT_COUNT):
        return
    s = _slots(uid)[slot]
    if image is not None:
        s["image"] = _clone_value("image", image)
        logging.info("[PersistentOutput] 槽%d 存图: %s", slot + 1, _mem_brief("image", image, s["image"]))
    if audio is not None:
        s["audio"] = _clone_value("audio", audio)
        logging.info("[PersistentOutput] 槽%d 存音: %s", slot + 1, _mem_brief("audio", audio, s["audio"]))


def _resolve_cache_target(cache, uid: str):
    """子图节点要先解析到所属层级缓存；顶层节点返回 cache 本身。"""
    resolve = getattr(cache, "_get_cache_for", None)
    if callable(resolve):
        try:
            resolved = resolve(uid)
            if resolved is not None:
                return resolved
        except Exception as e:
            logging.warning("[PersistentOutput] 解析缓存层级失败 uid=%s: %s", uid, e)
    return cache


def _evict_cache_entries(uid: str, keys) -> int:
    """从内核 outputs 缓存里摘掉指定 data key 的条目（镜像内核删除动作）。

    回放场景下条目里是存储副本——只清 _STORE 不清缓存，副本仍被缓存攥着，
    释放就不是真的；桥接场景下条目里是上游原件，但上游自己的缓存条目仍持有
    同一对象，摘我们的条目不损失任何数据。运行中途释放的极端情况：下游可能
    拿到 None——与"未保留输出 None"同语义。返回实际摘掉的条目数。
    """
    cache = _CACHE_HANDLES.get(uid)
    if cache is None or not keys:
        return 0
    removed = 0
    try:
        target = _resolve_cache_target(cache, uid)
        store = getattr(target, "cache", None)
        if not isinstance(store, dict):
            return 0
        for key in keys:
            if key not in store:
                continue
            del store[key]
            removed += 1
            # 镜像内核自己的删除：LRU/RAM 缓存的 side dict 不清会留残键
            for side in ("used_generation", "timestamps", "children"):
                sd = getattr(target, side, None)
                if isinstance(sd, dict):
                    sd.pop(key, None)
    except Exception as e:
        logging.warning("[PersistentOutput] 摘除输出缓存条目失败 uid=%s keys=%d: %s",
                        uid, len(tuple(keys)), e)
    return removed


def _record_cache_key(uid: str, out_cache) -> None:
    """execute 时调用：存句柄 + 记账本轮 data key + 清掉以前各轮的旧缓存条目。

    本轮的条目要等 execute 返回后才由内核写入（execution.py:616-618），因此
    此刻缓存里凡是我们记过账的 key 都属于以前的轮次；gen widget 保证本节点
    每轮必重跑，旧条目不可能再被命中，是纯死重，直接摘除。
    """
    if out_cache is None:
        return
    _CACHE_HANDLES[uid] = out_cache
    try:
        target = _resolve_cache_target(out_cache, uid)
        key_set = getattr(target, "cache_key_set", None)
        key = key_set.get_data_key(uid) if key_set is not None else None
    except Exception as e:
        logging.warning("[PersistentOutput] 记录缓存键失败 uid=%s: %s", uid, e)
        return
    if key is None:
        return
    keys = _CACHE_KEYS.setdefault(uid, set())
    stale = keys - {key}
    if stale:
        removed = _evict_cache_entries(uid, stale)
        logging.info("[PersistentOutput] 清理旧轮缓存条目 %d 条 uid=%s", removed, uid)
        keys -= stale
    keys.add(key)


def release(uid: str, target: str = "all") -> bool:
    """手动释放。target: "slot1"/"slot2"/"slot3" / "all"。返回是否确有内容被释放。

    释放 = 清 _STORE 的副本 + 摘本节点自己的输出缓存条目；上游节点自己的缓存
    不碰（原件由 ComfyUI 的缓存策略管辖）。三槽全空则整个键删掉。
    """
    if uid is None or uid not in _STORE:
        return False
    if target == "all":
        dropped = sum(_slot_bytes(s) for s in _STORE[uid])
        del _STORE[uid]
        had_any = True
        key_gone = True
        logging.info("[PersistentOutput] 释放 all: 移除副本字节=%s", f"{dropped:,}")
    else:
        idx = {"slot1": 0, "slot2": 1, "slot3": 2}.get(target)
        if idx is None:
            return False
        slots = _STORE.get(uid)
        if slots is None:
            return False
        had_any = slots[idx]["image"] is not None or slots[idx]["audio"] is not None
        logging.info("[PersistentOutput] 释放 %s: 移除副本字节=%s",
                     target, f"{_slot_bytes(slots[idx]):,}")
        slots[idx] = {"image": None, "audio": None}
        # 三槽全空就把整个键删掉，不留空壳
        key_gone = all(s["image"] is None and s["audio"] is None for s in slots)
        if key_gone:
            _STORE.pop(uid, None)
    if had_any:
        removed = _evict_cache_entries(uid, _CACHE_KEYS.get(uid, ()))
        _CACHE_KEYS.pop(uid, None)
        logging.info("[PersistentOutput] 释放时摘除自身缓存条目 %d 条 uid=%s", removed, uid)
        # 释放改变了槽状态：递增代数让下一轮缓存签名必变（见 execute 的 gen widget），
        # 防止缓存命中重放旧 ui 快照把前端盖暗。
        _bump_generation(uid)
    if key_gone:
        # 键已删，句柄失去存在意义（下次 execute 会重新存），一并清掉
        _CACHE_HANDLES.pop(uid, None)
    return had_any


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
            # 缓存命中时核心会重放上一轮的 ui 快照（emit_cached_output，execution.py:449）——
            # 槽状态在此期间被 release 改过的话，前端就被旧快照盖暗（用户实测踩过：
            # 亮→暗→亮）。声明 not_idempotent 把 unique_id 纳入缓存签名还不够；
            # 真正的对策是让缓存签名里带上"存储代数"：release 递增代数 → 签名变 →
            # 必然 miss 重跑 → ui 新鲜。见 _bump_generation 与 execute 的 selection 注入。
            not_idempotent=True,
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
                            tooltip=f"槽 {i} 的图像批次（视频帧）。有输入则替换保留并输出存储副本；断开后输出该槽保留内容。",
                        ),
                        io.Audio.Input(
                            f"audio_{i}",
                            display_name=f"音频 {i}",
                            optional=True,
                            tooltip=f"槽 {i} 的音频。有输入则替换保留并输出存储副本；断开后输出该槽保留内容。",
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
                io.String.Input(
                    "gen",
                    default="",
                    optional=True,
                    tooltip=(
                        "存储代数（内部防缓存机制）：release 会递增它，使下一轮缓存签名"
                        "必然变化、节点重跑并产出新鲜 ui（否则缓存命中会重放上一轮的旧"
                        "ui 快照，把前端状态盖暗）。由前端 serializeValue 写入，不要手填。"
                    ),
                ),
            ],
            # execution_list 必须显式声明：内核只对声明了的 hidden 注入执行列表
            # （execution.py:213-214），HiddenHolder 对未声明的 hidden 返回 None——
            # 漏了它释放路由就拿不到 outputs 缓存句柄，摘除永远空转（实测踩坑）。
            hidden=[io.Hidden.unique_id, io.Hidden.execution_list],
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
                image_3=None, audio_3=None, labels_json="[]", gen=""):
        uid = cls.hidden.unique_id
        if uid is None:
            raise ValueError(
                "PersistentOutput 拿不到节点 unique_id（仅在 ComfyUI 运行时可用）；"
                "请通过 ComfyUI 正常排队运行。"
            )
        del labels_json   # UI-only，后端不消费；形参仅防前端脚本未生效时报多余 kwarg

        # 临时探针：确认本轮 execute 真的在跑（缓存命中时根本不会进到这里）。
        logging.info(
            "[PersistentOutput] execute uid=%s gen_in=%r 有输入=%s",
            uid, gen, [img is not None or aud is not None for img, aud in
                       ((image_1, audio_1), (image_2, audio_2), (image_3, audio_3))],
        )

        inputs = [(image_1, audio_1), (image_2, audio_2), (image_3, audio_3)]
        outs = []
        for i, (img, aud) in enumerate(inputs):
            store_input(uid, i, image=img, audio=aud)
            slot = _slots(uid)[i]
            # 桥接 + 回放统一为「输出恒等于存储副本」：有输入时 store_input 已把
            # 深拷贝副本放进槽里，输出副本而不是上游原件——所有权隔离，绝不让我们
            # 的缓存条目 pin 住上游对象（见模块 docstring）。没输入出保留，都没有出 None。
            outs.extend([slot["image"], slot["audio"]])
        # ui 必须有：executed 事件只在本节点有 ui 输出时才发送（execution.py:567），
        # 没有它前端 onExecuted 永远不触发、状态行永远不刷新（用户实测踩过）。
        # 把各槽概要直接随事件送出去，前端免二次 HTTP。
        # 同时记账 outputs 缓存：释放路由没有 hidden 上下文，靠 hidden.execution_list
        # 拿句柄、记 data key，才能摘除本节点自己的缓存条目（回放副本才会真正释放）。
        el = getattr(cls.hidden, "execution_list", None)
        out_cache = getattr(el, "output_cache", None) if el is not None else None
        _record_cache_key(uid, out_cache)
        # ui 值一律包一层 list：execution.py:422 合并多个 ui 时按 key 把值当可迭代
        # 展开（{k: [y for x in uis for y in x[k]]}），裸标量（int/str）会直接炸
        # "'int' object is not iterable"（实测踩过）。
        return io.NodeOutput(*outs, ui={"psl_slots": describe(uid), "psl_gen": [_GENERATION.get(uid, 0)]})


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
        return web.json_response({"released": had, "slots": describe(uid), "gen": _GENERATION.get(uid, 0)})

