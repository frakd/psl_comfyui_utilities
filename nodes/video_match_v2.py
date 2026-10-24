"""Advanced Video Match V2 (PSL) — 对齐上一段视频末尾与下一段视频开头的色差。

从 SVI_with_PainterI2V 的 SVI_AdvancedVideoMatchV2 移植为 V3 节点，算法保持一致；
默认参数改用实际工作流里调好的那一组（window_size=3, cdf_weight=0.50,
cross_channel=0.10, cdf_max_shift=0.08, cdf_chroma_guard=0.90）。

相比原版 AdvancedVideoMatchNode 的核心改进:
1. 感知均匀色空间 (Oklab 默认 / Lab / srgb 对照), 在感知空间做匹配可避免 RGB 逐通道匹配的色相偏移;
2. 参考侧多帧稳健锚点: 取上段视频尾部 K 帧, 按帧间一致性加权, 抗单帧崩坏;
3. 3x3 全协方差线性 Monge-Kantorovich 映射 (闭式解, 零训练):
   保留逐通道 mean/std 匹配能力, 并加入交叉通道相关项, 修正"通道耦合"型偏色;
   交叉通道强度 (cross_channel) 可调, 降低可避免蓝→青这类色相旋转;
4. 每通道 CDF 直方图匹配 (1D 单调 LUT, 无空箱/无带状), 修正非线性色差 (黑位/伽马);
   带偏移上限 (cdf_max_shift) 与饱和像素色度保护 (cdf_chroma_guard),
   防止紫色/红色等饱和色被推到锚点色度尾部而变深变艳;
5. 双锚点目标插值 (stat_mode=dual_anchor, 默认):
   锚点 A = v1 尾部基准, 锚点 B = v2 自身稳定基准 (尾段);
   每帧目标基准 = p·A + (1-p)·B 随余弦强度曲线平滑插值, 逐帧用稳定的
   每通道均值/方差映射 (无协方差/LUT 噪声), 帧间无统计漂移跳变,
   从 v1 色调基准平滑过渡回 v2 自身色调; frozen (固定变换) 与
   rolling (逐帧 EMA) 模式保留供对照;
6. 保留原版的时间窗口 / 强度衰减 / 高光保护 / 亮度自适应逻辑。

新增 stable_anchor 模式 (自动稳健路线):
- 场景切分对所有 stat_mode 生效: 参考 A 只取参考视频最后一个镜头段,
  目标 B/修正只取目标开头第一个镜头段, 修正到目标第一个硬切处停止;
  切镜后的画面既不参与锚点, 也不会继续被接缝映射改色;
- 接缝局域锚点: 参考侧只在末尾 max(2×ref_anchor_frames,16) 帧里筛稳定帧,
  取最后 K 帧; 目标侧只在开头同一窗口里筛稳定帧;
- 用一条 B→A 的固定映射按强度曲线混合, 保留目标片段内部的曝光/对比动态,
  不再逐帧归一化 (这是 dual_anchor 在大范围明度变化上的根因);
- 自动明度软上限: 限制每帧平均明度的净偏移, 避免参考/目标基准差过大时整段变亮变暗;
- 低对比置信门控: 溶解/闪白帧自动降低修正强度, 避免 scale 打满造成的饱和度尖峰;
- 低色度保护: 肤色/中性色只保留部分 a/b 修正, 防止参考偏色把皮肤推成洋红;
- 瞬态过曝平滑: 用相邻帧亮度识别开头 1-2 帧的正向尖峰, 再对亮度曲线做高斯时域平滑,
  把峰的能量摊到前后帧; 硬切/淡出处分段, 不跨切镜扩散;
  首帧是否按闪光处理由上一段视频末帧亮度判断, 接缝切镜/转场不处理;
- cross_channel 在 stable_anchor 下通过 _lk_map 真正生效。

输出语义与原版一致: 参考视频 + 修正后的目标视频, 可直接替换对比。
"""

import math

import torch
import torch.nn.functional as F
from comfy_api.latest import io

# stable_anchor 的稳健性阈值 (颜色分量按其取值范围归一化)
_STABLE_JUMP = 0.02          # 相邻帧平均色变化超过 2% 视为跳变/淡出
_STABLE_LOW_CONTRAST = 0.35  # 对比度低于中位数的 35% 视为溶解/闪白/黑场
_STABLE_OUTLIER = 0.10       # 稳定帧中位色偏离超过 10% 视为离群
_STABLE_CONF_LO = 0.25       # 对比度低于锚点的 25% 时完全跳过修正
_STABLE_CONF_SPAN = 0.50     # 25%..75% 之间线性过渡到全强度
_LUMA_SHIFT_FRACTION = 0.10  # 每帧平均明度向下偏移上限 (占 L 通道范围的 10%)
_LUMA_RISE_FRACTION = 0.03   # 向上提亮上限更紧, 避免整帧高亮
# 低色度区域 (肤色/中性色) 的色相最敏感: cross_channel 与逐通道 CDF 容易把
# 肤色推向洋红。对 a/b 修正做自动回拉, 只保留部分匹配; L 仍完整匹配。
_LOW_CHROMA_GUARD_LO = 0.03
_LOW_CHROMA_GUARD_HI = 0.20
_LOW_CHROMA_GUARD_STRENGTH = 0.80
# 瞬态过曝压制: 用相邻帧亮度的中位数作为局部基线, 只处理正向尖峰;
# 检测到峰后对亮度曲线做高斯时域平滑, 把峰值能量摊到前后帧。
_FLASH_RADIUS = 4
_FLASH_L_LO = 0.004          # 占 L 通道范围的比例
_FLASH_CUT_DL = 0.040        # 切镜判定: 跳变前后窗口中位数差超过该比例
_FLASH_CUT_RADIUS = 5        # 切镜检测窗口, 大于闪光长度, 避免把 1-2 帧闪光当切镜
_SCENE_CUT_DIFF = 0.08       # 去均值缩略图 MAD 超过该值视为场景切镜
_SCENE_MIN_STD = 0.02        # 低对比/溶解帧不参与切镜判定
_SCENE_THUMB = 32            # 场景切镜检测用的缩略图边长
_FLASH_SMOOTH_RADIUS = 4
_FLASH_SMOOTH_SIGMA = 1.6
_FLASH_STRENGTH = 0.90


class PSLAdvancedVideoMatchV2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PSLAdvancedVideoMatchV2",
            display_name="Advanced Video Match V2 (PSL)",
            category="video/psl",
            search_aliases=[
                "video",
                "color",
                "match",
                "色差",
                "调色",
                "拼接",
                "对齐",
                "镜头匹配",
            ],
            description=(
                "对齐上一段视频末尾与下一段视频开头的色差。默认 dual_anchor 模式在 "
                "Oklab 空间按参考视频尾部多帧锚点做逐帧稳定映射，强度随余弦曲线平滑"
                "衰减回目标视频自身色调，另可选 frozen / rolling 对照。"
                "输出为参考帧 + 修正后的目标帧拼接。"
            ),
            inputs=[
                io.Image.Input(
                    "reference_video",
                    tooltip="上一段视频（取尾部 ref_anchor_frames 帧作为颜色锚点）。",
                ),
                io.Image.Input(
                    "target_video",
                    tooltip="下一段视频，逐帧向参考视频的色调对齐。",
                ),
                io.Int.Input(
                    "window_size",
                    default=3,
                    min=0,
                    max=200,
                    tooltip="前几帧全强度匹配",
                ),
                io.Int.Input(
                    "fade_out",
                    default=20,
                    min=1,
                    max=500,
                    tooltip="强度线性衰减到 0 的帧位置",
                ),
                io.Float.Input(
                    "match_intensity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    tooltip="全局匹配强度",
                ),
                io.Float.Input(
                    "highlight_protect",
                    default=0.85,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="高光保护阈值 (越大保护越强)",
                ),
                io.Int.Input(
                    "ref_anchor_frames",
                    default=8,
                    min=1,
                    max=48,
                    tooltip=(
                        "参考视频尾部锚点帧数 (多帧加权, 抗单帧崩坏); "
                        "stable_anchor 下为参考侧稳定窗口的最大帧数"
                    ),
                ),
                io.Float.Input(
                    "stat_smoothing",
                    default=0.85,
                    min=0.0,
                    max=0.95,
                    step=0.01,
                    tooltip="rolling 模式的逐帧统计 EMA 平滑系数 (其他模式不生效)",
                ),
                io.Combo.Input(
                    "stat_mode",
                    options=["dual_anchor", "stable_anchor", "frozen", "rolling"],
                    default="dual_anchor",
                    tooltip=(
                        "变换模式: dual_anchor=双锚点逐帧统计(旧默认); "
                        "stable_anchor=接缝局域稳定锚点+场景切分+固定映射+自动明度上限"
                        "+溶解帧门控+低色度肤色保护+瞬态过曝时域平滑(切镜/接缝保护, "
                        "推荐用于有曝光变化/淡出/溶解的片段); "
                        "frozen=固定变换全窗口复用; rolling=逐帧EMA更新(旧版行为)"
                    ),
                ),
                io.Combo.Input(
                    "color_space",
                    options=["oklab", "lab", "srgb"],
                    default="oklab",
                    tooltip="感知均匀色空间; srgb 为对照 (同旧节点逻辑)",
                ),
                io.Combo.Input(
                    "match_mode",
                    options=["affine_cdf", "affine_only", "cdf_only"],
                    default="affine_cdf",
                    tooltip=(
                        "affine_cdf=全协方差矩阵+直方图; affine_only=仅矩阵; "
                        "cdf_only=仅直方图"
                    ),
                ),
                io.Float.Input(
                    "cdf_weight",
                    default=0.50,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="affine_cdf 模式下直方图修正的权重",
                ),
                io.Float.Input(
                    "cross_channel",
                    default=0.10,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "3x3 映射中交叉通道项强度: 0=纯逐通道(无旋转, 旧节点同族), "
                        "1=全协方差。蓝→青这类色相漂移调低它。"
                        "stable_anchor/frozen/rolling 生效; dual_anchor 为逐通道映射, 此参数无效"
                    ),
                ),
                io.Float.Input(
                    "cdf_max_shift",
                    default=0.08,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip=(
                        "CDF 每通道偏移上限(占通道范围比例)。防止饱和色被推到锚点"
                        "色度尾部而变深变艳"
                    ),
                ),
                io.Float.Input(
                    "cdf_chroma_guard",
                    default=0.90,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip="对已饱和像素的 a/b 通道修正抑制力度(0=关闭), 防紫色/红色过饱和",
                ),
            ],
            outputs=[io.Image.Output("image")],
        )

    # ---------------------------------------------------------------
    # 色空间转换 (输入输出均为非线性 sRGB [0,1])
    # ---------------------------------------------------------------
    @staticmethod
    def _srgb_to_linear(c):
        c = c.clamp(0, 1)
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def _linear_to_srgb(c):
        return torch.where(c <= 0.0031308, c * 12.92,
                           1.055 * torch.sign(c) * torch.abs(c).pow(1.0 / 2.4) - 0.055)

    @staticmethod
    def _cbrt(t):
        return torch.sign(t) * torch.abs(t).pow(1.0 / 3.0)

    @classmethod
    def _space_fns(cls, name, device):
        """返回 (to_space_fn, from_space_fn, 每通道取值范围) """
        if name == "srgb":
            return (lambda x: x, lambda x: x, [(0.0, 1.0)] * 3)

        if name == "lab":
            M = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]], device=device)
            Minv = torch.tensor([
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252]], device=device)
            delta = 6.0 / 29.0
            delta3 = delta ** 3

            def lab_f(t):
                return torch.where(t > delta3, cls._cbrt(t),
                                   t / (3.0 * delta * delta) + 4.0 / 29.0)

            def inv_f(t):
                return torch.where(t > delta,
                                   t ** 3,
                                   3.0 * delta * delta * (t - 4.0 / 29.0))

            def to_space(x):
                lin = cls._srgb_to_linear(x)
                xyz = (lin @ M.t()).clamp(0)
                fx = lab_f(xyz[..., 0:1] / 0.95047)
                fy = lab_f(xyz[..., 1:2] / 1.0)
                fz = lab_f(xyz[..., 2:3] / 1.08883)
                L = 116.0 * fy - 16.0
                a = 500.0 * (fx - fy)
                b = 200.0 * (fy - fz)
                return torch.cat([L, a, b], dim=-1)

            def from_space(y):
                L, a, b = y[..., 0:1], y[..., 1:2], y[..., 2:3]
                fy = (L + 16.0) / 116.0
                fx = fy + a / 500.0
                fz = fy - b / 200.0
                X = 0.95047 * inv_f(fx)
                Y = 1.0 * inv_f(fy)
                Z = 1.08883 * inv_f(fz)
                lin = torch.cat([X, Y, Z], dim=-1) @ Minv.t()
                return cls._linear_to_srgb(lin)

            return (to_space, from_space, [(0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0)])

        # Oklab
        M1 = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]], device=device)
        M2 = torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]], device=device)
        M1i = torch.tensor([
            [1.0, 0.3963377774, 0.2158037573],
            [1.0, -0.1055613458, -0.0638541728],
            [1.0, -0.0894841775, -1.2914855480]], device=device)
        M2i = torch.tensor([
            [4.0767416621, -3.3077115913, 0.2309699292],
            [-1.2684380046, 2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147, 1.7076147010]], device=device)

        def to_space(x):
            lin = cls._srgb_to_linear(x)
            lms = lin @ M1.t()
            return cls._cbrt(lms) @ M2.t()

        def from_space(y):
            lms_ = y @ M1i.t()   # (L, a, b) -> (l', m', s')
            lms = lms_ ** 3
            lin = lms @ M2i.t()  # (l, m, s) -> 线性 sRGB
            return cls._linear_to_srgb(lin)

        return (to_space, from_space, [(0.0, 1.0), (-0.6, 0.6), (-0.6, 0.6)])

    # ---------------------------------------------------------------
    # 统计工具
    # ---------------------------------------------------------------
    @classmethod
    def _fit_size(cls, H, W, target=256):
        s = target / max(H, W)
        return max(16, int(H * s)), max(16, int(W * s))

    @classmethod
    def _downscale(cls, img, size_hw):
        # img: [H, W, 3] -> [h, w, 3] (area 采样, 统计用)
        hs, ws = size_hw
        p = img.permute(2, 0, 1).unsqueeze(0).float()
        return F.interpolate(p, size=(hs, ws), mode='area').squeeze(0).permute(1, 2, 0)

    @classmethod
    def _clip_stats(cls, video, fit, to_space, device):
        """逐帧目标色空间均值/标准差 (降采样统计), 供 stable_anchor 筛选。"""
        mus, sigs = [], []
        for f in video:
            sm = to_space(cls._downscale(f.to(device), fit)).reshape(-1, 3)
            mus.append(sm.mean(0))
            sigs.append(sm.std(0))
        return torch.stack(mus), torch.stack(sigs)

    @classmethod
    def _stable_indices(cls, mus, sigs, scales):
        """筛掉低对比/跳变/离群帧；返回 (索引, 是否回退到全部帧)。"""
        if mus.shape[0] == 0:
            return [], True
        contrast = sigs.norm(dim=1)
        ok = contrast > _STABLE_LOW_CONTRAST * torch.median(contrast)
        if mus.shape[0] > 1:
            jumps = ((mus[1:] - mus[:-1]) / scales).pow(2).sum(dim=-1).sqrt()
            unstable = jumps > _STABLE_JUMP
            ok[1:] &= ~unstable
            ok[:-1] &= ~unstable
        # 离群判据只对已通过对比度/跳变的帧生效, 否则长淡出会把正常帧判成离群
        if int(ok.sum()) >= 2:
            med = torch.median(mus[ok], dim=0).values
            dev = ((mus - med) / scales).pow(2).sum(dim=-1).sqrt()
            ok &= dev < _STABLE_OUTLIER
        idx = torch.nonzero(ok, as_tuple=False).flatten().tolist()
        if len(idx) >= 2:
            return idx, False
        return list(range(mus.shape[0])), True

    @classmethod
    def _scene_cuts(cls, video, device):
        """基于去均值缩略图的相邻帧 MAD 检测硬切; 对全局曝光变化不敏感。"""
        n = video.shape[0]
        if n < 2:
            return torch.zeros(0, dtype=torch.bool, device=device)
        h, w = video.shape[1], video.shape[2]
        hs, ws = cls._fit_size(h, w, target=_SCENE_THUMB)
        prev = None
        prev_std = 0.0
        flags = []
        for frame in video:
            thumb = cls._downscale(frame.to(device), (hs, ws))
            thumb = thumb - thumb.mean(dim=(0, 1), keepdim=True)
            thumb_std = float(thumb.std())
            if prev is not None:
                stable = prev_std > _SCENE_MIN_STD and thumb_std > _SCENE_MIN_STD
                flags.append(stable and float((thumb - prev).abs().mean()) > _SCENE_CUT_DIFF)
            prev, prev_std = thumb, thumb_std
        return torch.tensor(flags, device=device, dtype=torch.bool)

    @classmethod
    def _cut_flags(cls, values, cut_threshold, radius=_FLASH_CUT_RADIUS):
        """用跳变前后窗口中位数的差判断硬切; 短促闪光会回到原电平, 不会被判成切镜。"""
        n = values.shape[0]
        if n < 2:
            return torch.zeros(0, dtype=torch.bool, device=values.device)
        vals = values.detach().cpu().tolist()
        cuts = []
        for k in range(n - 1):
            before = vals[max(0, k - radius):k + 1]
            after = vals[k + 1:min(n, k + 1 + radius)]
            if not before or not after:
                cuts.append(False)
                continue
            # 切镜: 前后窗口的 min 和 max 同向移动; 闪光只抬高/压低一侧
            d_max = max(after) - max(before)
            d_min = min(after) - min(before)
            cuts.append((d_max > cut_threshold and d_min > cut_threshold) or
                        (d_max < -cut_threshold and d_min < -cut_threshold))
        return torch.tensor(cuts, device=values.device, dtype=torch.bool)

    @classmethod
    def _temporal_baseline(cls, values, cuts, stability_threshold, radius=_FLASH_RADIUS):
        """双侧中位数基线: 只有同时高于前后两侧的短促凸起才算过曝尖峰。

        首帧用后侧中位数判断 (目标片段可能只有 3 帧, 开头亮峰不能漏);
        但如果后侧窗口内存在硬切, 或后侧本身是淡出/曝光斜坡, 则跳过首帧。
        末帧没有后侧参考, 跳过。切点附近的帧也不做峰检测。
        """
        n = values.shape[0]
        if n < 2:
            return values.clone()
        vals = values.detach().cpu().tolist()
        cut_list = cuts.detach().cpu().tolist() if torch.is_tensor(cuts) else list(cuts)
        base = list(vals)
        if n >= 3 and not any(cut_list[:min(radius, n - 1)]):
            following = vals[1:min(n, 1 + radius)]
            # 后侧窗口若仍是单调淡出/曝光斜坡, 不当作首帧闪光
            if following and max(following) - min(following) <= stability_threshold:
                base[0] = sorted(following)[len(following) // 2]
        for i in range(1, n - 1):
            lo, hi = max(0, i - radius), min(n, i + radius + 1)
            if any(cut_list[lo:hi - 1]):
                base[i] = vals[i]
                continue
            before = sorted(vals[lo:i])
            after = sorted(vals[i + 1:hi])
            left = before[len(before) // 2] if before else vals[i]
            right = after[len(after) // 2] if after else vals[i]
            base[i] = max(left, right)
        return torch.tensor(base, device=values.device, dtype=values.dtype)

    @classmethod
    def _first_frame_is_seam(cls, first, reference_last, threshold):
        """首帧不高于上一段末帧 -> 接缝切镜/转场, 不按本段闪光处理。"""
        return first <= reference_last + threshold

    @classmethod
    def _gaussian_smooth_segment(cls, values, radius=_FLASH_SMOOTH_RADIUS,
                                 sigma=_FLASH_SMOOTH_SIGMA):
        x = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
        kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        padded = F.pad(values.view(1, 1, -1), (radius, radius))
        counts = F.pad(torch.ones_like(values).view(1, 1, -1), (radius, radius))
        numerator = F.conv1d(padded, kernel.view(1, 1, -1))
        denominator = F.conv1d(counts, kernel.view(1, 1, -1))
        return (numerator / denominator.clamp_min(1e-12)).view(-1)

    @classmethod
    def _gaussian_smooth(cls, values, cuts=None, radius=_FLASH_SMOOTH_RADIUS,
                         sigma=_FLASH_SMOOTH_SIGMA):
        """在切点处分段平滑, 避免高斯核跨过切镜把两边混在一起。"""
        n = values.shape[0]
        if n == 0:
            return values.clone()
        if cuts is None or not bool(cuts.any()):
            return cls._gaussian_smooth_segment(values, radius, sigma)
        out = values.clone()
        start = 0
        for end in range(n):
            if end == n - 1 or bool(cuts[end]):
                out[start:end + 1] = cls._gaussian_smooth_segment(
                    values[start:end + 1], radius, sigma)
                start = end + 1
        return out

    @classmethod
    def _peak_mask(cls, excess, threshold, cuts=None, radius=_FLASH_SMOOTH_RADIUS):
        mask = torch.zeros_like(excess)
        n = excess.shape[0]
        for i in torch.nonzero(excess > threshold, as_tuple=False).flatten().tolist():
            mask[i] = 1.0
            if cuts is None:
                mask[max(0, i - radius):min(n, i + radius + 1)] = 1.0
                continue
            for j in range(i - 1, max(-1, i - radius - 1), -1):
                if bool(cuts[j]):
                    break
                mask[j] = 1.0
            for j in range(i + 1, min(n, i + radius + 1)):
                if bool(cuts[j - 1]):
                    break
                mask[j] = 1.0
        return mask

    @classmethod
    def _anchor_stats(cls, video, indices, fit, to_space, device):
        mus, covs = [], []
        for t in indices:
            sm = to_space(cls._downscale(video[t].to(device), fit)).reshape(-1, 3)
            mu = sm.mean(0)
            cen = sm - mu
            mus.append(mu)
            covs.append((cen.t() @ cen) / max(1, sm.shape[0] - 1))
        return torch.stack(mus).mean(0), torch.stack(covs).mean(0)

    @classmethod
    def _anchor_hist(cls, video, indices, fit, to_space, device, ranges, nbins):
        hists = []
        for t in indices:
            sm = to_space(cls._downscale(video[t].to(device), fit)).reshape(-1, 3)
            hists.append(cls._histogram(sm, ranges, nbins))
        return torch.stack(hists).mean(0)

    @classmethod
    def _luma_signal(cls, x_space, color_space, device):
        if color_space == "srgb":
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
            return (x_space * weights).sum(dim=-1)
        return x_space[..., 0]

    @classmethod
    def _cap_luma_shift(cls, x_space, out, color_space, cap_down, cap_up, device):
        """只压整帧平均明度的净偏移, 保留局部对比与逐帧曝光动态。"""
        shift = cls._luma_signal(out, color_space, device).mean() - \
            cls._luma_signal(x_space, color_space, device).mean()
        cap = cap_up if float(shift) > 0 else cap_down
        if float(shift.abs()) <= cap:
            return out
        excess = shift - cap * torch.tanh(shift / cap)
        if color_space == "srgb":
            out = out - excess
        else:
            out[..., 0] -= excess
        return out

    @classmethod
    def _histogram(cls, flat, ranges, nbins=512):
        # flat: [N, 3] -> [3, nbins] (带边界 clamp 的加权直方图)
        h = []
        for c in range(3):
            lo, hi = ranges[c]
            v = flat[:, c].clamp(lo, hi)
            idx = ((v - lo) / (hi - lo) * (nbins - 1)).long()
            h.append(torch.bincount(idx, minlength=nbins).float())
        return torch.stack(h)

    @classmethod
    def _build_lut(cls, src_hist, tgt_cdf, nbins=512, max_shift_bins=None):
        # 1D 单调 LUT: 目标箱索引 (直方图匹配, 无空箱问题)
        # max_shift_bins: 每箱相对恒等映射的最大偏移 (防饱和色被推到锚点色度尾部)
        src_cdf = src_hist.cumsum(1)
        src_cdf = src_cdf / src_cdf[:, -1:].clamp_min(1e-12)
        lut = torch.zeros_like(src_cdf).long()
        for c in range(3):
            # 只允许在锚点实际占用的范围内映射, 避免空箱被拉到通道极值
            pos = (tgt_cdf[c] > 0.0).nonzero(as_tuple=True)[0]
            lo_b, hi_b = 0, nbins - 1
            if pos.numel() > 0:
                lo_b, hi_b = int(pos[0]), int(pos[-1])
            j = torch.searchsorted(tgt_cdf[c], src_cdf[c]).clamp(lo_b, hi_b)
            j = torch.cummax(j, 0).values
            if max_shift_bins is not None and max_shift_bins < nbins:
                i = torch.arange(nbins, device=j.device)
                j = i + (j - i).clamp(-max_shift_bins, max_shift_bins)
            lut[c] = torch.cummax(j, 0).values
        return lut

    @classmethod
    def _chroma_guards(cls, xin, ranges, strength, device):
        # 色度保护: 越饱和的像素, a/b 通道的 CDF 修正越弱 (防止紫色/红色被加深)
        g = torch.ones_like(xin)
        if strength <= 1e-5:
            return g
        mid1 = 0.5 * (ranges[1][0] + ranges[1][1])
        mid2 = 0.5 * (ranges[2][0] + ranges[2][1])
        scale = max(0.5 * (ranges[1][1] - ranges[1][0]),
                    0.5 * (ranges[2][1] - ranges[2][0]), 1e-6)
        c = torch.sqrt((xin[:, 1] - mid1) ** 2 + (xin[:, 2] - mid2) ** 2) / scale
        t = ((c - 0.15) / (0.60 - 0.15)).clamp(0, 1)
        g[:, 1:] = (1.0 - float(strength) * t).unsqueeze(-1)
        return g

    @classmethod
    def _hist_has_spread(cls, hist, min_var_bins=4.0):
        # 直方图有效散布检查: 防止恒定帧 (delta 分布) 的 CDF 匹配病态退化
        p = hist / hist.sum(1, keepdim=True).clamp_min(1e-12)
        idx = torch.arange(hist.shape[1], device=hist.device, dtype=p.dtype)
        mu = (p * idx).sum(1)
        var = (p * (idx - mu[:, None]) ** 2).sum(1)
        return float(var.max()) >= min_var_bins

    @classmethod
    def _apply_lut(cls, flat, lut, ranges, nbins=512):
        # 带线性插值的 LUT 应用 (插值消除量化带状)
        out = torch.empty_like(flat)
        for c in range(3):
            lo, hi = ranges[c]
            v = flat[:, c].clamp(lo, hi)
            pos = (v - lo) / (hi - lo) * (nbins - 1)
            i0 = pos.floor().long().clamp(0, nbins - 1)
            i1 = (i0 + 1).clamp(0, nbins - 1)
            frac = (pos - i0.float()).unsqueeze(-1)
            j = lut[c][i0].float().unsqueeze(-1) * (1.0 - frac) + lut[c][i1].float().unsqueeze(-1) * frac
            out[:, c] = lo + j.squeeze(-1) * (hi - lo) / (nbins - 1)
        return out

    @classmethod
    def _lk_map(cls, mu_s, cov_s, mu_t, cov_t, cross_channel=1.0):
        """全协方差线性 Monge-Kantorovich: A = Ss^-1/2 (Ss^1/2 St Ss^1/2)^1/2 Ss^-1/2, b = mu_t - A mu_s
        对角协方差时退化为逐通道 scale/offset, 因此是原算法的严格泛化。
        cross_channel < 1 时把 A 的交叉通道项按比例削弱 (降低色相旋转风险);
        奇异值钳制防止极端放大 (防爆炸, 默认宽松)。"""
        def eigh_clamped(m):
            vals, vecs = torch.linalg.eigh((m + m.t()) / 2.0)
            vals = vals.clamp_min(vals.max().clamp_min(1e-12) * 1e-3)
            return vals, vecs

        def sym_sqrt(m):
            vals, vecs = eigh_clamped(m)
            return (vecs * vals.sqrt()) @ vecs.t()

        def sym_inv_sqrt(m):
            vals, vecs = eigh_clamped(m)
            return (vecs * (1.0 / vals.sqrt())) @ vecs.t()

        mu_s64, cov_s64 = mu_s.double(), cov_s.double()
        mu_t64, cov_t64 = mu_t.double(), cov_t.double()

        # 退化保护: 任一侧方差近似为 0 (平坦帧) 时, 退化为纯均值修正, 避免逆平方根爆炸
        lam_s = float(torch.linalg.eigvalsh(cov_s64).max())
        lam_t = float(torch.linalg.eigvalsh(cov_t64).max())
        if lam_s < 1e-8 or lam_t < 1e-8:
            return torch.eye(3, dtype=mu_s.dtype, device=mu_s.device), (mu_t64 - mu_s64).float()

        S_s = sym_sqrt(cov_s64)
        S_s_inv = sym_inv_sqrt(cov_s64)
        M = sym_sqrt(S_s @ cov_t64 @ S_s)
        A = S_s_inv @ M @ S_s_inv
        # 交叉通道强度: 0 = 逐通道参考映射 (无旋转, 旧节点同族), 1 = 全协方差 LK
        if cross_channel < 1.0:
            s_pc = torch.sqrt(torch.clamp(torch.diagonal(cov_t64), min=1e-12) /
                              torch.clamp(torch.diagonal(cov_s64), min=1e-12))
            A_pc = torch.diag(s_pc)
            A = A_pc + float(cross_channel) * (A - A_pc)
        # 安全钳制: 奇异值增益 [1/2.5, 2.5], 防极端放大
        U, S, Vh = torch.linalg.svd(A)
        S = S.clamp(1.0 / 2.5, 2.5)
        A = (U * S) @ Vh
        b = mu_t64 - A @ mu_s64
        return A.float(), b.float()

    # ---------------------------------------------------------------
    # 主流程
    # ---------------------------------------------------------------
    @classmethod
    def execute(cls, reference_video, target_video, window_size, fade_out,
                match_intensity, highlight_protect, ref_anchor_frames,
                stat_smoothing, stat_mode, color_space, match_mode,
                cdf_weight, cross_channel, cdf_max_shift, cdf_chroma_guard):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = reference_video.float()
        vid = target_video.float()
        B, H, W, _ = vid.shape
        if B == 0:
            return io.NodeOutput(ref)
        if ref.shape[0] == 0:
            raise ValueError("reference_video is empty")
        if ref.shape[1:3] != vid.shape[1:3]:
            raise ValueError(
                "reference_video and target_video must share H/W; "
                f"got {tuple(ref.shape[1:3])} vs {tuple(vid.shape[1:3])}"
            )

        to_space, from_space, ranges = cls._space_fns(str(color_space), device)
        nbins = 512
        max_shift_bins = int(round(nbins * float(cdf_max_shift))) if float(cdf_max_shift) < 1.0 else None
        mode = str(stat_mode)
        use_dual = mode == "dual_anchor"
        use_stable = mode == "stable_anchor"
        use_frozen = mode == "frozen"

        # 场景切分对所有模式生效: 色调/亮度匹配只应发生在同一镜头内
        ref_cuts = cls._scene_cuts(ref, device)
        tgt_cuts = cls._scene_cuts(vid, device)
        ref_scene_start = 0
        for k in range(0, ref.shape[0] - 1):
            if bool(ref_cuts[k]):
                ref_scene_start = k + 1
        target_cut_frame = None
        for k in range(0, vid.shape[0] - 1):
            if bool(tgt_cuts[k]):
                target_cut_frame = k
                break

        # ---------- 1. 参考侧: 最后一个镜头段内的尾部 K 帧 ----------
        K = max(1, min(int(ref_anchor_frames), ref.shape[0]))
        fit = cls._fit_size(H, W)
        ref_anchor_start = max(ref.shape[0] - K, ref_scene_start)
        frame_mus, frame_covs, frame_hists = [], [], []
        for t in range(ref_anchor_start, ref.shape[0]):
            sm = to_space(cls._downscale(ref[t].to(device), fit)).reshape(-1, 3)
            mu = sm.mean(0)
            cen = sm - mu
            cov = (cen.t() @ cen) / max(1, sm.shape[0] - 1)
            frame_mus.append(mu)
            frame_covs.append(cov)
            frame_hists.append(cls._histogram(sm, ranges, nbins))
        frame_mus = torch.stack(frame_mus)
        frame_covs = torch.stack(frame_covs)
        frame_hists = torch.stack(frame_hists)

        # 帧间一致性加权: 离中位数越远的帧 (崩坏帧) 权重越低
        med = torch.median(frame_mus, dim=0).values
        scales = torch.tensor([r[1] - r[0] for r in ranges], device=device).clamp_min(1e-3)
        d = ((frame_mus - med) / scales).pow(2).sum(dim=-1).sqrt()
        wts = torch.exp(-(d ** 2) / (2.0 * 0.05 ** 2))
        wts = wts / (wts.sum() + 1e-12)
        mu_t = (frame_mus * wts[:, None]).sum(0)
        cov_t = (frame_covs * wts[:, None, None]).sum(0)
        hist_t = (frame_hists * wts[:, None, None]).sum(0)
        cdf_t = hist_t.cumsum(1)
        cdf_t = cdf_t / cdf_t[:, -1:].clamp_min(1e-12)

        if not use_stable:
            print(
                f"[PSL V2] anchors ref={ref_anchor_start}..{ref.shape[0] - 1} "
                f"cut={target_cut_frame} space={color_space} mode={match_mode}"
            )

        # ---------- 2. 目标侧: 变换与强度 ----------
        alpha = max(0.0, min(0.95, float(stat_smoothing)))
        window = int(window_size)
        fade = max(int(fade_out), window + 1)

        # ---------- 2.0 stable_anchor: 接缝局域稳定锚点 + 固定映射 ----------
        # A = 参考视频尾部窗口内的稳定帧; B = 目标视频开头窗口内的稳定帧。
        # 只取接缝附近, 避免整段视频的全局平均把开头映射到别的场景/曝光上。
        stable_state = None
        if use_stable:
            ref_mus, ref_sigs = cls._clip_stats(ref, fit, to_space, device)
            tgt_mus, tgt_sigs = cls._clip_stats(vid, fit, to_space, device)
            ref_luma = cls._luma_signal(ref_mus.unsqueeze(0), str(color_space), device).squeeze(0)
            tgt_luma = cls._luma_signal(tgt_mus.unsqueeze(0), str(color_space), device).squeeze(0)
            tail_n = min(3, ref_luma.shape[0])
            ref_seam_luma = float(torch.median(ref_luma[-tail_n:]))

            anchor_window = max(2 * K, 16)
            ref_start = max(0, ref.shape[0] - anchor_window)
            ref_seg_start = max(ref_start, ref_scene_start)
            ref_local_idx, ref_fallback = cls._stable_indices(
                ref_mus[ref_seg_start:], ref_sigs[ref_seg_start:], scales)
            ref_idx = [ref_seg_start + i for i in ref_local_idx]
            ia = ref_idx[-K:]

            tgt_head_end = min(vid.shape[0], anchor_window)
            if target_cut_frame is not None:
                tgt_head_end = min(tgt_head_end, target_cut_frame + 1)
            tgt_idx, tgt_fallback = cls._stable_indices(
                tgt_mus[:tgt_head_end], tgt_sigs[:tgt_head_end], scales)

            cut_frame = target_cut_frame

            max_b = max(2 * K, 16)
            if len(tgt_idx) > max_b:
                ib = [tgt_idx[int(i * len(tgt_idx) / max_b)] for i in range(max_b)]
            else:
                ib = tgt_idx

            mu_t, cov_t = cls._anchor_stats(ref, ia, fit, to_space, device)
            mu_b, cov_b = cls._anchor_stats(vid, ib, fit, to_space, device)
            sig_b = cov_b.diagonal().clamp_min(1e-4).sqrt()
            A, b = cls._lk_map(mu_b, cov_b, mu_t, cov_t, float(cross_channel))

            hist_t = cls._anchor_hist(ref, ia, fit, to_space, device, ranges, nbins)
            cdf_t = hist_t.cumsum(1)
            cdf_t = cdf_t / cdf_t[:, -1:].clamp_min(1e-12)
            lut = None
            if match_mode == "cdf_only":
                hist_src = cls._anchor_hist(vid, ib, fit, to_space, device, ranges, nbins)
                if cls._hist_has_spread(hist_src):
                    lut = cls._build_lut(hist_src, cdf_t, nbins, max_shift_bins)
            elif match_mode == "affine_cdf" and float(cdf_weight) > 1e-5:
                hists = []
                for t in ib:
                    sm = to_space(cls._downscale(vid[t].to(device), fit)).reshape(-1, 3)
                    hists.append(cls._histogram(sm @ A.t() + b, ranges, nbins))
                hist_src = torch.stack(hists).mean(0)
                if cls._hist_has_spread(hist_src):
                    lut = cls._build_lut(hist_src, cdf_t, nbins, max_shift_bins)

            stable_state = (A, b, lut, sig_b, tgt_luma, ref_seam_luma, cut_frame, tgt_cuts)
            if ref_fallback or tgt_fallback:
                print(
                    "[PSL V2] stable_anchor: not enough stable frames in seam window, "
                    f"fallback ref={ref_fallback} tgt={tgt_fallback}"
                )

            def _span(idx):
                return f"{idx[0]}..{idx[-1]}(n={len(idx)})" if idx else "none"

            print(
                f"[PSL V2] stable anchors ref={_span(ia)} tgt={_span(ib)} "
                f"window={anchor_window} cut={cut_frame} "
                f"scene_cuts ref={int(ref_cuts.sum())} tgt={int(tgt_cuts.sum())} "
                f"refL={mu_t[0]:.3f} tgtL={mu_b[0]:.3f} "
                f"cross={cross_channel} fit={fit} space={color_space}"
            )

        # ---------- 2.1 dual_anchor: 目标开头第一场景段尾帧作为 B ----------
        # 逐帧目标基准 = p·A + (1-p)·B, 随强度曲线平滑插值:
        #   p=1 (首帧) -> v1 尾部基准 (衔接处完全对齐);
        #   p=0 (稳定段) -> v2 自身基准 (原始本色);
        # 切镜处会强制 strength=0, 不会把映射带进下一场景。
        dual_state = None
        if use_dual:
            # B 只取目标开头第一个镜头段的尾帧, 不再从整段目标尾部取
            tgt_scene_end = (target_cut_frame + 1) if target_cut_frame is not None else B
            K2 = max(1, min(int(ref_anchor_frames), tgt_scene_end))
            b_start = max(0, tgt_scene_end - K2)
            b_mus, b_covs, b_hists = [], [], []
            for t in range(b_start, tgt_scene_end):
                sm = to_space(cls._downscale(vid[t].to(device), fit)).reshape(-1, 3)
                mu = sm.mean(0)
                cen = sm - mu
                cov = (cen.t() @ cen) / max(1, sm.shape[0] - 1)
                b_mus.append(mu)
                b_covs.append(cov)
                b_hists.append(cls._histogram(sm, ranges, nbins))
            b_mus = torch.stack(b_mus)
            b_covs = torch.stack(b_covs)
            b_hists = torch.stack(b_hists)
            med_b = torch.median(b_mus, dim=0).values
            d_b = ((b_mus - med_b) / scales).pow(2).sum(dim=-1).sqrt()
            wts_b = torch.exp(-(d_b ** 2) / (2.0 * 0.05 ** 2))
            wts_b = wts_b / (wts_b.sum() + 1e-12)
            mu_b = (b_mus * wts_b[:, None]).sum(0)
            cov_b = (b_covs * wts_b[:, None, None]).sum(0)
            hist_b = (b_hists * wts_b[:, None, None]).sum(0)
            cdf_b = hist_b.cumsum(1)
            cdf_b = cdf_b / cdf_b[:, -1:].clamp_min(1e-12)
            sig_a = cov_t.diagonal().clamp_min(1e-4).sqrt()
            sig_b = cov_b.diagonal().clamp_min(1e-4).sqrt()

            # 冻结形状 LUT 对 (基于第 0 帧, 分别朝 A / B 两个目标构建)
            x0 = vid[0].to(device)
            xs0 = to_space(cls._downscale(x0, fit)).reshape(-1, 3)
            mu0 = xs0.mean(0)
            sig0 = xs0.std(0).clamp_min(1e-4)
            h0 = cls._histogram(xs0, ranges, nbins)
            lut_aa = lut_ba = lut_ar = lut_br = None
            if match_mode == "cdf_only":
                if cls._hist_has_spread(h0):
                    lut_ar = cls._build_lut(h0, cdf_t, nbins, max_shift_bins)
                    lut_br = cls._build_lut(h0, cdf_b, nbins, max_shift_bins)
            elif float(cdf_weight) > 1e-5:
                y0a = (xs0 - mu0) * (sig_a / sig0) + mu_t
                y0b = (xs0 - mu0) * (sig_b / sig0) + mu_b
                h_a = cls._histogram(y0a, ranges, nbins)
                h_b = cls._histogram(y0b, ranges, nbins)
                if cls._hist_has_spread(h_a):
                    lut_aa = cls._build_lut(h_a, cdf_t, nbins, max_shift_bins)
                if cls._hist_has_spread(h_b):
                    lut_ba = cls._build_lut(h_b, cdf_b, nbins, max_shift_bins)
            dual_state = (mu_t, sig_a, mu_b, sig_b, lut_aa, lut_ba, lut_ar, lut_br)

        # ---------- 2.2 frozen: 用第 0 帧统计预计算一次变换 (A/b/LUT), 全窗口复用 ----------
        # 帧 0 的修正结果与旧行为完全一致 (旧行为第 0 帧也只用自己的统计)。
        frozen = None
        if use_frozen:
            x0 = vid[0].to(device)
            xs0 = to_space(cls._downscale(x0, fit)).reshape(-1, 3)
            mu_s0 = xs0.mean(0)
            cen0 = xs0 - mu_s0
            cov_s0 = (cen0.t() @ cen0) / max(1, xs0.shape[0] - 1)
            hist0 = cls._histogram(xs0, ranges, nbins)
            A0, b0 = cls._lk_map(mu_s0, cov_s0, mu_t, cov_t, float(cross_channel))
            lut0 = None
            if match_mode == "cdf_only":
                if cls._hist_has_spread(hist0):
                    lut0 = cls._build_lut(hist0, cdf_t, nbins, max_shift_bins)
            elif match_mode == "affine_cdf" and float(cdf_weight) > 1e-5:
                hist_aff0 = cls._histogram(xs0 @ A0.t() + b0, ranges, nbins)
                if cls._hist_has_spread(hist_aff0):
                    lut0 = cls._build_lut(hist_aff0, cdf_t, nbins, max_shift_bins)
            frozen = (A0, b0, lut0)

        ema_mean = ema_cov = ema_hist_raw = ema_hist_aff = None
        corrected = []
        out_luma = []

        for i in range(B):
            x = vid[i].to(device)

            # 平滑强度曲线: 前 window 帧全强度, 之后余弦衰减到 fade_out。
            # 余弦曲线在两端导数为 0, 避免线性衰减的"拐点感", 过渡连续平滑。
            if i < window:
                p = 1.0
            elif i >= fade:
                p = 0.0
            else:
                t = (i - window) / max(1, fade - window)
                p = 0.5 * (1.0 + math.cos(math.pi * t))
            strength = p * float(match_intensity)
            if target_cut_frame is not None and i > target_cut_frame:
                strength = 0.0

            if strength <= 1e-5:
                corrected.append(x.cpu())
                if stable_state is not None:
                    out_luma.append(float(stable_state[4][i]))
                continue

            x_space = to_space(x)
            xi = x_space.reshape(-1, 3)

            if stable_state is not None:
                # stable_anchor: 固定映射 + 低对比置信门控 (溶解/闪白帧自动减力)
                A, b, lut, sig_b, tgt_luma, _, _, _ = stable_state
                xs_space = to_space(cls._downscale(x, fit)).reshape(-1, 3)
                sig_s = xs_space.std(0).clamp_min(1e-4)
                ratio = (sig_s / sig_b).min().clamp(0, 1)
                conf = float(((ratio - _STABLE_CONF_LO) / _STABLE_CONF_SPAN).clamp(0, 1))
                strength *= conf

                y_aff = xi @ A.t() + b
                if match_mode == "cdf_only":
                    y = xi
                    if lut is not None:
                        y_lut = cls._apply_lut(xi, lut, ranges, nbins)
                        guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                        y = xi + (y_lut - xi) * guards
                else:
                    y = y_aff
                    if match_mode == "affine_cdf" and lut is not None:
                        y_cdf = cls._apply_lut(y_aff, lut, ranges, nbins)
                        guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                        y = y_aff + float(cdf_weight) * (y_cdf - y_aff) * guards

                # 低色度保护: 肤色/中性色只保留部分 a/b 修正, 防止色相被推成洋红
                c_in = torch.sqrt(xi[:, 1] ** 2 + xi[:, 2] ** 2)
                chroma_w = ((c_in - _LOW_CHROMA_GUARD_LO) /
                            (_LOW_CHROMA_GUARD_HI - _LOW_CHROMA_GUARD_LO)).clamp(0.0, 1.0)
                protect = (1.0 - chroma_w) * _LOW_CHROMA_GUARD_STRENGTH
                y[:, 1:] = xi[:, 1:] + (y[:, 1:] - xi[:, 1:]) * (1.0 - protect[:, None])
            elif dual_state is not None:
                # dual_anchor: 逐帧稳定的每通道均值/方差映射, 目标基准随 p 平滑插值
                xs_space = to_space(cls._downscale(x, fit)).reshape(-1, 3)
                mu_s = xs_space.mean(0)
                sig_s = xs_space.std(0).clamp_min(1e-4)
                mu_a, sig_av, mu_b, sig_bv = dual_state[0], dual_state[1], dual_state[2], dual_state[3]
                mu_T = mu_a * p + mu_b * (1.0 - p)
                sig_T = sig_av * p + sig_bv * (1.0 - p)
                scale = (sig_T / sig_s).clamp(0.25, 4.0)
                offs = mu_T - scale * mu_s

                if match_mode == "cdf_only":
                    lut_ar, lut_br = dual_state[6], dual_state[7]
                    lut = None
                    if lut_ar is not None and lut_br is not None:
                        lut = p * lut_ar + (1.0 - p) * lut_br
                    else:
                        lut = lut_ar if lut_ar is not None else lut_br
                    y = xi
                    if lut is not None:
                        y_lut = cls._apply_lut(xi, lut, ranges, nbins)
                        guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                        y = xi + (y_lut - xi) * guards
                else:
                    y_aff = xi * scale + offs
                    y = y_aff
                    if match_mode == "affine_cdf" and float(cdf_weight) > 1e-5:
                        lut_aa, lut_ba = dual_state[4], dual_state[5]
                        lut = None
                        if lut_aa is not None and lut_ba is not None:
                            lut = p * lut_aa + (1.0 - p) * lut_ba
                        else:
                            lut = lut_aa if lut_aa is not None else lut_ba
                        if lut is not None:
                            y_cdf = cls._apply_lut(y_aff, lut, ranges, nbins)
                            guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                            y = y_aff + float(cdf_weight) * (y_cdf - y_aff) * guards
            elif frozen is not None:
                # frozen: 固定变换, 全窗口复用 (帧间无漂移跳变)
                A, b, lut = frozen
                y = xi @ A.t() + b
                if match_mode == "cdf_only":
                    if lut is not None:
                        y_lut = cls._apply_lut(xi, lut, ranges, nbins)
                        guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                        y = xi + (y_lut - xi) * guards
                elif lut is not None:  # affine_cdf
                    y_aff = y
                    y_cdf = cls._apply_lut(y_aff, lut, ranges, nbins)
                    guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                    y = y_aff + float(cdf_weight) * (y_cdf - y_aff) * guards
            else:
                # rolling: 逐帧 EMA 统计 + 每帧变换 (旧版行为)
                xs_space = to_space(cls._downscale(x, fit)).reshape(-1, 3)
                mu_s = xs_space.mean(0)
                cen = xs_space - mu_s
                cov_s = (cen.t() @ cen) / max(1, xs_space.shape[0] - 1)
                hist_raw = cls._histogram(xs_space, ranges, nbins)

                if ema_mean is None:
                    ema_mean, ema_cov, ema_hist_raw = mu_s, cov_s, hist_raw
                else:
                    ema_mean = alpha * ema_mean + (1.0 - alpha) * mu_s
                    ema_cov = alpha * ema_cov + (1.0 - alpha) * cov_s
                    ema_hist_raw = alpha * ema_hist_raw + (1.0 - alpha) * hist_raw

                if match_mode == "affine_only":
                    A, b = cls._lk_map(ema_mean, ema_cov, mu_t, cov_t, float(cross_channel))
                    y = xi @ A.t() + b
                elif match_mode == "cdf_only":
                    if cls._hist_has_spread(ema_hist_raw):
                        lut = cls._build_lut(ema_hist_raw, cdf_t, nbins, max_shift_bins)
                        y_lut = cls._apply_lut(xi, lut, ranges, nbins)
                        guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                        y = xi + (y_lut - xi) * guards
                    else:
                        A, b = cls._lk_map(ema_mean, ema_cov, mu_t, cov_t, float(cross_channel))
                        y = xi @ A.t() + b
                else:  # affine_cdf: 先线性映射, 再在映射结果上做直方图形状修正
                    A, b = cls._lk_map(ema_mean, ema_cov, mu_t, cov_t, float(cross_channel))
                    y_aff = xi @ A.t() + b
                    y = y_aff
                    if float(cdf_weight) > 1e-5:
                        y_aff_small = xs_space @ A.t() + b
                        hist_aff = cls._histogram(y_aff_small, ranges, nbins)
                        if ema_hist_aff is None:
                            ema_hist_aff = hist_aff
                        else:
                            ema_hist_aff = alpha * ema_hist_aff + (1.0 - alpha) * hist_aff
                        if cls._hist_has_spread(ema_hist_aff):
                            lut = cls._build_lut(ema_hist_aff, cdf_t, nbins, max_shift_bins)
                            y_cdf = cls._apply_lut(y_aff, lut, ranges, nbins)
                            guards = cls._chroma_guards(xi, ranges, float(cdf_chroma_guard), device)
                            y = y_aff + float(cdf_weight) * (y_cdf - y_aff) * guards

            delta = (y - xi).reshape(H, W, 3)

            # --- 高光保护 + 亮度自适应 (与原节点一致) ---
            luma = x.max(dim=-1, keepdim=True).values
            prot = ((1.0 - luma) / (1.0 - float(highlight_protect) + 1e-5)).clamp(0, 1)
            adaptive = 1.0 + ((luma - luma.mean()).clamp(0, 1) * 0.2)
            final_strength = (strength * prot * adaptive).to(x_space.dtype)

            out = x_space + delta * final_strength
            if stable_state is not None:
                # 保持输入帧的 L 对比度: 禁止 A/B 的 std 差把整段对比度压缩/拉高
                if str(color_space) != "srgb":
                    l_in = x_space[..., 0]
                    l_out = out[..., 0]
                    in_std = l_in.std()
                    out_std = l_out.std().clamp_min(1e-6)
                    ratio = (in_std / out_std).clamp(0.5, 2.0)
                    out[..., 0] = l_out.mean() + (l_out - l_out.mean()) * ratio

                l_range = ranges[0][1] - ranges[0][0]
                out = cls._cap_luma_shift(
                    x_space, out, str(color_space),
                    _LUMA_SHIFT_FRACTION * l_range,
                    _LUMA_RISE_FRACTION * l_range,
                    device,
                )
                out_luma.append(float(cls._luma_signal(out, str(color_space), device).mean()))
            corrected.append(from_space(out).clamp(0, 1).cpu())

        # 瞬态过曝: 在输出亮度曲线上检测短促正向尖峰, 用高斯时域平滑把峰摊到前后帧
        if stable_state is not None and float(match_intensity) > 0 and len(out_luma) == B:
            luma_curve = torch.tensor(out_luma, device=device)
            l_scale = ranges[0][1] - ranges[0][0]
            cut_threshold = _FLASH_CUT_DL * l_scale
            cuts = cls._cut_flags(luma_curve, cut_threshold) | stable_state[7]
            baseline = cls._temporal_baseline(luma_curve, cuts, cut_threshold)
            if cls._first_frame_is_seam(float(luma_curve[0]), stable_state[5],
                                        _FLASH_L_LO * l_scale):
                baseline[0] = luma_curve[0]
            mask = cls._peak_mask(luma_curve - baseline, _FLASH_L_LO * l_scale, cuts)
            if bool(mask.any()):
                smoothed = cls._gaussian_smooth(luma_curve, cuts)
                delta_l = (smoothed - luma_curve) * mask * _FLASH_STRENGTH
                for i in torch.nonzero(delta_l.abs() > 1e-6, as_tuple=False).flatten().tolist():
                    o = to_space(corrected[i].to(device))
                    if str(color_space) == "srgb":
                        o = o + float(delta_l[i])
                    else:
                        o[..., 0] += float(delta_l[i])
                    corrected[i] = from_space(o).clamp(0, 1).cpu()

        result = torch.stack(corrected, dim=0)
        return io.NodeOutput(torch.cat([ref, result], dim=0))
