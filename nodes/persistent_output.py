"""Persistent Output — 跨运行保留输入的桥接/回放节点（3 组「图像+音频」槽位）。

设计给「对比多轮视频结果」用：每组槽位是图像批次+音频的一对，规则相同且相互独立：

- 本轮该槽有输入：存储替换为新输入，输出 = 输入（纯桥接）
- 本轮该槽无输入（上游断开或上游给了 None）：输出 = 该槽上一次保留的内容
- 该槽从未保留过：输出 None（下游收到 None，等价于没连线；下游需容忍 None）

典型对比流：轮 1 接上游跑（槽 1 保留+桥接）→ 改参数跑轮 2（槽 2）→ 再跑轮 3（槽 3）
→ 断开上游：槽 1/槽 2 的输出分别接 A/B Compare 两耳，与实时结果并排对比。

内存策略（定稿哲学：**私有副本 A 只属于 _STORE，永远不出节点**）：
- 存入时 clone 一份私有副本 A 进 _STORE；输出**永远不是 A**——桥接给上游原件
  （内核自己的对象，随它 pin），回放给 A 的临时克隆。无论下游消费了什么、内核
  条目引用了什么，都 pin 不到 A 头上：release 清 _STORE 的瞬间 A 必然无人引用、
  内存立刻回落（判决性不变量，用户定稿的"两份副本"设计）。
- 内核会对每个节点的输出**无条件**写运行缓存（execution.py:616-618）。条目里的
  是**引用**（不额外占内存），但引用会让对象活着。只干预**我们自己节点**的条目，
  精确按 key：① execute 时记账当轮 key、摘掉历轮旧条目（它引用着上一轮交给
  内核的输出——不摘它，旧输出永远活着）；② release 时摘掉自己的全部条目。
  本轮条目必须活到本轮结束——下游节点取上游数据走的就是这份缓存（graph.py
  get_local），当轮摘了下游拿到 None。
- 释放 = 清 _STORE（A 立刻消失，无条件）+ ②的摘除（上一轮输出的去向随缘）。
  剩余例外（接受，不处理）：交出去的输出被下游消费后，下游条目的引用会 pin 到
  内核压力驱逐——内核对所有节点的统一行为，/psl/persist/cache_debug 端点可对账；
  上游 VHS 自己条目里的原件同理。需要全部立即清掉时用前端自带的「释放内存」
  按钮（官方 reset）。
- gen widget 按需自增（仅释放后）：内容没变时签名不变，内核条目同 key 替换不
  留尸体；释放后必 miss 重放释放后的新状态。
- 其余不变：只存内存不落盘，进程退出即释放；键是节点 unique_id；三槽全空
  删整个键；节点从画布删除时前端自动全清。
- 代价：桥接期间内容在内存里存两份（私有 A + 上游缓存原件），回放期间存两份
  （A + 临时克隆）——这是"A 永不出节点"的必要成本；换来 release 的无条件生效。
"""

import asyncio
import gc
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


# unique_id -> 本节点最近一次执行时的内核 outputs 缓存句柄与当轮 data key。
# 句柄/键用于：execute 时摘掉历轮旧条目（旧副本尸体）、release 时摘掉自己的
# 条目与引用了副本的下游条目。**本轮条目绝不能在当轮摘除**——下游节点取上游
# 数据走的就是同一份缓存（graph.py get_local），当轮摘了下游会拿到 None。
_CACHE_HANDLE: dict[str, object] = {}
_CACHE_KEYS: dict[str, set] = {}


def _tensor_ptrs(value, seen: set | None = None) -> set[int]:
    """递归收集输出里的 CPU 张量 data_ptr（内容指纹，用于诊断对账）。"""
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.numel() > 0:
            seen.add(value.untyped_storage().data_ptr())
    elif isinstance(value, dict):
        for v in value.values():
            _tensor_ptrs(v, seen)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _tensor_ptrs(v, seen)
    return seen


def _entry_stats(outputs, seen_storages: set | None = None) -> tuple[int, set[int]]:
    """统计一个 CacheEntry.outputs 的 CPU 张量字节数与内容指纹（镜像内核 ram_release 的扫描）。"""
    total = 0
    ptrs: set[int] = set()
    seen = seen_storages if seen_storages is not None else set()

    def walk(v):
        nonlocal total
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, torch.Tensor):
            if v.device.type == "cpu" and v.numel() > 0:
                key = (v.untyped_storage().data_ptr(), v.untyped_storage().nbytes())
                if key not in seen:
                    seen.add(key)
                    total += v.untyped_storage().nbytes()
                    ptrs.add(v.untyped_storage().data_ptr())

    walk(outputs)
    return total, ptrs


def _evict_keys(cache, uid: str, keys) -> int:
    """从内核 outputs 缓存摘除指定 data key 的条目（镜像内核删除动作，含侧表）。"""
    if cache is None or not keys:
        return 0
    store = getattr(cache, "cache", None)
    if not isinstance(store, dict):
        return 0
    removed = 0
    try:
        for key in list(keys):
            if key not in store:
                continue
            del store[key]
            removed += 1
            # 镜像内核自己的删除：LRU/RAM 缓存的 side dict 不清会留残键
            for side in ("used_generation", "timestamps", "children"):
                sd = getattr(cache, side, None)
                if isinstance(sd, dict):
                    sd.pop(key, None)
    except Exception as e:
        logging.warning("[PersistentOutput] 摘除缓存条目失败 uid=%s: %s", uid, e)
    return removed


def _prune_stale_cache_entries(uid: str, out_cache) -> None:
    """execute 时调用：记账本轮 data key，并摘掉以前各轮的旧条目。

    本轮条目要等 execute 返回后才由内核写入（execution.py:616-618），因此此刻
    缓存里凡是我们记过账的 key 都属于以前的轮次。gen 按需自增后签名只在内容
    变化/释放后改变，但内容一旦变化旧条目就是攥着旧副本的尸体（旧副本已从
    _STORE 被替换），摘掉才能让旧副本立刻回收。本轮条目**必须保留**——下游
    节点取上游数据走的就是这份缓存（graph.py get_local）。
    """
    if out_cache is None:
        return
    _CACHE_HANDLE[uid] = out_cache
    try:
        key_set = getattr(out_cache, "cache_key_set", None)
        key = key_set.get_data_key(uid) if key_set is not None else None
    except Exception as e:
        logging.warning("[PersistentOutput] 记录缓存键失败 uid=%s: %s", uid, e)
        return
    if key is None:
        return
    keys = _CACHE_KEYS.setdefault(uid, set())
    stale = keys - {key}
    if stale:
        removed = _evict_keys(out_cache, uid, stale)
        logging.info("[PersistentOutput] 清理旧轮缓存条目 %d 条 uid=%s", removed, uid)
        keys -= stale
    keys.add(key)


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


def release(uid: str, target: str = "all") -> bool:
    """手动释放。target: "slot1"/"slot2"/"slot3" / "all"。返回是否确有内容被释放。

    释放 = 清 _STORE 的副本 + 摘本节点历轮输出缓存条目（它们引用着副本，不清
    则对象继续存活、内存不降）。上游/下游节点的缓存条目不碰——内核对所有节点
    的统一行为；副本若已被下游消费，其条目的引用会 pin 到内核压力驱逐（诊断
    端点 /psl/persist/cache_debug 可见）。三槽全空则整个键删掉。运行中途释放的
    极端情况：下游本轮可能拿到 None——与"未保留输出 None"同语义。
    """
    if uid is None or uid not in _STORE:
        return False
    if target == "all":
        victims = list(_STORE[uid])
        slot_idx = None
    else:
        idx = {"slot1": 0, "slot2": 1, "slot3": 2}.get(target)
        if idx is None:
            return False
        victims = [_STORE[uid][idx]]
        slot_idx = idx
    had_any = any(s["image"] is not None or s["audio"] is not None for s in victims)
    dropped = sum(_slot_bytes(s) for s in victims)
    # 引用者诊断：副本理应只有 _STORE 一处引用；若 release 后 RSS 不降，
    # 这里打印的名字就是真凶。
    import sys
    for s in victims:
        v = s.get("image")
        if isinstance(v, torch.Tensor):
            referrers = gc.get_referrers(v)
            ref_desc = [f"{type(r).__name__}({len(r) if hasattr(r, '__len__') else '?'})"
                        for r in referrers]
            logging.info("[PersistentOutput] 副本引用者诊断: refcount=%d 引用者=%s",
                         sys.getrefcount(v), ref_desc)

    if slot_idx is None:
        del _STORE[uid]
    else:
        slots = _STORE[uid]
        slots[slot_idx] = {"image": None, "audio": None}
        # 三槽全空就把整个键删掉，不留空壳
        if all(s["image"] is None and s["audio"] is None for s in slots):
            _STORE.pop(uid, None)

    if had_any:
        removed_own = _evict_keys(_CACHE_HANDLE.get(uid), uid, _CACHE_KEYS.get(uid, ()))
        _CACHE_KEYS.pop(uid, None)
        logging.info("[PersistentOutput] 释放 %s: 副本字节=%s 摘自身缓存条目=%d",
                     target, f"{dropped:,}", removed_own)
        # 释放改变了槽状态：递增代数让下一轮缓存签名必变（gen widget 按需自增），
        # 下一轮重放的是释放后的新状态，不会命中释放前的旧输出。
        _bump_generation(uid)
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
            # （execution.py:213-214）。摘除本轮输出缓存条目全靠它的 output_cache 句柄。
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
            # 输出**永不使用私有副本 A**（_slots 里的那份）：桥接给上游原件（内核
            # 自己的对象，随它 pin），回放给 A 的临时克隆——A 从不离开本节点，
            # release 清 _STORE 时 A 必然无人引用、立刻回收（判决性不变量）。
            outs.extend([
                img if img is not None else _clone_value("image", slot["image"]),
                aud if aud is not None else _clone_value("audio", slot["audio"]),
            ])
        # ui 必须有：executed 事件只在本节点有 ui 输出时才发送（execution.py:567），
        # 没有它前端 onExecuted 永远不触发、状态行永远不刷新（用户实测踩过）。
        # 把各槽概要直接随事件送出去，前端免二次 HTTP。
        # 同时记账 outputs 缓存：内核必然会把我们的输出写进缓存条目
        # （execution.py:616-618），这里记账当轮 key 并摘掉历轮旧条目（旧副本
        # 尸体），释放路由才能按账摘除。本轮条目保留到当轮结束。
        el = getattr(cls.hidden, "execution_list", None)
        out_cache = getattr(el, "output_cache", None) if el is not None else None
        _prune_stale_cache_entries(uid, out_cache)
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


def _mem_snapshot() -> str:
    """系统内存快照（与任务管理器口径对照：系统可用≈内存Available，
    进程RSS≈工作集，进程USS≈进程独占）。psutil 缺失时给出明确提示。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        proc = psutil.Process()
        parts = [f"系统可用={vm.available / 1024 ** 3:.2f}GB",
                 f"进程RSS={proc.memory_info().rss / 1024 ** 3:.2f}GB"]
        try:
            parts.append(f"进程USS={proc.memory_full_info().uss / 1024 ** 3:.2f}GB")
        except Exception:
            pass
        return " ".join(parts)
    except Exception as e:
        return f"内存监控不可用: {e}"


if getattr(PromptServer, "instance", None) is not None:

    @PromptServer.instance.routes.get("/psl/persist/status")
    async def persist_status(request):
        uid = _parse_uid(request.query.get("node_id"))
        if uid is None:
            return web.json_response({"error": "缺少 node_id"}, status=400)
        return web.json_response({"slots": describe(uid)})

    @PromptServer.instance.routes.get("/psl/persist/cache_debug")
    async def persist_cache_debug(request):
        """只读诊断：摊开内核 outputs 缓存每个条目的 CPU 张量字节数与内容指纹，
        以及 _STORE 的占用——"内存被谁攥着"一眼可见。"""
        uid = _parse_uid(request.query.get("node_id"))
        cache = _CACHE_HANDLE.get(uid) if uid else None
        if cache is None:
            return web.json_response({"error": "本节点尚未执行过（无缓存句柄）"}, status=404)
        store = getattr(cache, "cache", None)
        if not isinstance(store, dict):
            return web.json_response({"error": "缓存结构不符合预期"}, status=500)
        entries = []
        store_bytes = 0
        store_ptrs: set[int] = set()
        if uid in _STORE:
            for i, s in enumerate(_STORE[uid]):
                b, p = _entry_stats([s.get("image"), s.get("audio")])
                store_bytes += b
                store_ptrs |= p
        for i, (key, entry) in enumerate(store.items()):
            b, p = _entry_stats(getattr(entry, "outputs", None))
            entries.append({
                "index": i,
                "bytes": b,
                "touches_store": bool(p & store_ptrs),
                "ptrs": [hex(x) for x in p],
            })
        total = sum(e["bytes"] for e in entries)
        return web.json_response({
            "store_bytes": store_bytes,
            "store_slots_held": uid in _STORE,
            "kernel_entries": entries,
            "kernel_total_bytes": total,
        })

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
        mem_before = _mem_snapshot()
        had = release(uid, target)
        logging.info("[PersistentOutput] 释放 node=%s target=%s 有内容=%s", uid, target, had)
        logging.info("[PersistentOutput] 释放前内存: %s", mem_before)

        async def _report_after():
            await asyncio.sleep(3)
            gc.collect()
            logging.info("[PersistentOutput] 释放后3s内存: %s", _mem_snapshot())

        asyncio.create_task(_report_after())
        return web.json_response({"released": had, "slots": describe(uid), "gen": _GENERATION.get(uid, 0)})

