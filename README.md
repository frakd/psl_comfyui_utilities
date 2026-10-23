# psl_comfyui_utilities

个人用的 ComfyUI 自定义节点合集 / A personal collection of ComfyUI custom nodes.

所有节点均为 V3 节点（`comfy_api.latest` 的 `io.ComfyNode`），只依赖 ComfyUI 自带能力与 PyTorch，无额外第三方依赖。唯一例外：Video Combine (Memory) 需要系统 PATH 上有 ffmpeg（外部子进程，非 Python 依赖）。
All nodes are V3 nodes (`io.ComfyNode` from `comfy_api.latest`). They rely only on what ships with ComfyUI plus PyTorch — no extra third-party dependencies. The one exception: Video Combine (Memory) needs `ffmpeg` on the system PATH (an external subprocess, not a Python dependency).

---

## 安装 / Installation

把本目录放进 `ComfyUI/custom_nodes/` 后重启 ComfyUI。
Drop this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

在节点搜索框里输入 `psl` 或 `personal` 可以一次性筛出本包全部节点。
Typing `psl` or `personal` in the node search box filters out every node in this pack at once.

---

## 节点一览 / Node Overview

| 节点 Node | 一句话作用 / What it does |
| --- | --- |
| [Deep VRAM Cleanup](#deep-vram-cleanup) | 手动清显存/内存：清节点缓存、卸载模型、归还内存，节点上直接看诊断 |
| [Audio Harshness Suppressor](#audio-harshness-suppressor) | ⚠️ 实验性：只在真正刺耳的那几帧压高频，整条轨道不闷（效果尚未调好） |
| [Audio A/B Compare](#audio-ab-compare) | 两路音频合成一条左右分声道，戴耳机同时 A/B 对比 |
| [Any Switch](#any-switch) | 可视化多路开关，勾选第几路输出第几路，或每趟随机一路 |
| [MiniMax H3 AV LoRA Stack](#minimax-h3-av-lora-stack) | 给 H3 音视频模型叠 8 路 LoRA，每路各自调音频/视频权重 |
| [Aspect Ratio Selector](#aspect-ratio-selector) | 画布上拖矩形选宽高比，按总像素数算出整数 W×H |
| [Model Reroute (Step LoRA Switch)](#model-reroute-step-lora-switch) | 采样中途热切换 LoRA 分支，前段用 A 后段用 B |
| [LoRA Step Schedule (Bypass)](#lora-step-schedule-bypass) | ⚠️ 实验性：每一步给 LoRA 不同强度；曲线/手填列表 + 节点内可拖动曲线 |
| [Mask Paint Output](#mask-paint-output) | 预览底图、在其上画遮罩，成对输出 IMAGE + MASK |
| [Persistent Output](#persistent-output) | 三槽位记住前几趟运行的图/音频，断线后仍可复现做对比 |
| [Refreshable Text Box](#refreshable-text-box) | 文本框自动刷新显示上游文本，且不污染运行缓存 |
| [CLIP Text Encode (Cached)](#clip-text-encode-cached) | 文本编码结果按 (CLIP, 文本) 缓存，上游重跑但文本没变时不再重编码 |
| [Image Container (6 Outputs)](#image-container-6-outputs) | 图片池 + 点选排序（多选/单选/顺序单选），输出 6 路单图与 1 个 batch |
| [Video Combine (Memory)](#video-combine-memory) | 帧+音频直接在内存合成 MP4 不落盘（NVENC 加速引擎），合并/落盘接官方节点 |
| [Advanced Video Match V2 (PSL)](#advanced-video-match-v2-psl) | 对齐前后两段视频的色差：参考尾帧锚点 + Oklab 稳健匹配 |

---

## 节点说明 / Node Reference

### Deep VRAM Cleanup

`nodes/deep_vram_cleanup.py` · `ui/deep_vram_cleanup.js` · 分类 Category：`utils/VRAM`

**中** —— 手动内存/显存清扫节点。勾选开关决定它做哪些事：清全部节点缓存、只清指定来源节点的缓存、卸载模型（含连带断开的模型）、`empty_cache`、裁剪 CUDA 池、清理游离张量。节点上会显示诊断文本框，逐条写明清理前后的显存/内存与各动作的实际结果。它是输出节点，可以直接选中它单独按 Run，也可以串在链路里当透传。
注意三点：① `clear_source_node_cache` 开启时前端会自动让本节点每次强制重跑，否则命中缓存就被跳过、清理不生效；② 只连了来源节点但没开定向清理开关时是**干跑**，只在诊断里统计各来源能释放多少缓存，不删任何东西；③ `purge_orphan_tensors` 会遍历全部 gc 对象拆 CUDA 张量，重且可能影响其他节点的内部状态，不确定就别开。系统内存归还那部分依赖 Windows 工作集修剪，其他平台会自动跳过。

**EN** — A manual VRAM/RAM janitor node. Toggles decide what it actually does: clear all node cache, clear only selected source nodes' cache, unload models (optionally including disconnected ones), `empty_cache`, trim the CUDA pool, purge orphan tensors. An on-node diagnostics box reports VRAM/RAM before and after plus the outcome of each action. It's an output node, so you can run it alone or leave it inline as a passthrough.
Three things worth knowing: (1) when targeted cache clearing is on, the front end auto-increments a nonce so this node really re-runs — a cache hit would otherwise skip the side effects entirely; (2) connecting cache sources *without* enabling the clearing toggle gives you a **dry run** that only measures how much each source could release; (3) `purge_orphan_tensors` walks every gc-tracked object detaching CUDA tensors — slow, aggressive, and it can disturb other nodes' state, so leave it off unless you know why you need it. The RAM-return step uses a Windows working-set trim and is skipped on other platforms.

---

### Audio Harshness Suppressor

`nodes/audio_filter/` · 分类 Category：`audio/filter`

> ⚠️ **实验性节点，效果尚未调好 / Experimental — not tuned to a satisfactory result.**
> 目前没有可靠的调参办法：判别量在真实素材上分不开「刺耳段」与「正常人声」，默认参数取的是**不把事情弄坏**的一组，不是**能解决问题**的一组。
> No reliable tuning procedure exists yet — the decision value cannot separate genuinely harsh passages from ordinary vocals on real audio, so the defaults are chosen to avoid damage, not to fix the problem.

**中** —— 处理音频里扎耳朵的高频：齿音嘶声、镲片飞溅、过亮的"存在感"。它不是静态 EQ：逐帧比较目标频段与全谱的能量密度，只在频段真正突出的那些瞬间施加衰减，齿音之间的元音基本不动。因为判据是比值，与输入音量无关，`sensitivity_db` 不需要逐素材重标。可调频段上下限、触发阈值、最深衰减、attack/release 与干湿比。详细原理、默认值的实测依据和**已知失效模式**（非谐波伪影类问题不在能力范围内）见 [`nodes/audio_filter/README.md`](nodes/audio_filter/README.md)。

**EN** — Tames harsh high frequencies: sibilance, cymbal splash, over-bright "presence". It is deliberately not a static EQ: it measures band-vs-full-spectrum energy density per STFT frame and attenuates only in the moments where the band actually pokes out, so the material between the harsh hits stays put. The decision value is a ratio, hence level-independent, so `sensitivity_db` doesn't need recalibrating per file. Exposed: band low/high edges, trigger threshold, max reduction, attack/release, wet-dry mix. The sub-README documents the detection principle, how the defaults were measured, and a **known failure mode** (non-harmonic artefacts are out of scope).

---

### Audio A/B Compare

`nodes/ab_audio_compare.py` · 分类 Category：`audio/compare`

**中** —— 把两路音频合成一条左右分声道的对比音频：A 进左耳、B 进右耳，同一副耳机就能同时听两路做对比，不必来回切播放。核心已有的 `JoinAudioChannels` 在这个场景不合适（只收 mono、会把长边裁短、也不带预览），所以本节点自己实现：采样率不同时只把**较慢的一路上采样**到较快的一路（只升不降，不引入有损处理）；长度不同时短的一路**尾部补静音**而不裁剪——哪边先结束本身就是对比信息；多声道输入按均值下混；**不做响度匹配**，因为音量差也是对比对象。`swap_channels` 可以左右换边复听，用来排除人耳左右灵敏度差异带来的错觉。节点自带播放器，输出也能接 SaveAudio 存文件。

**EN** — Merges two audio signals into one split-channel file: A to the left ear, B to the right, so you can audition both at once on the same headphones instead of flipping between plays. The built-in `JoinAudioChannels` doesn't fit this job (mono-only, truncates the longer side, no preview), so this node does it itself: when sample rates differ only the **lower-rate side is upsampled** toward the higher one (never downsampled — lossy processing would contaminate the comparison); when lengths differ the shorter side is **padded with silence** rather than cropped, since which one ends first is itself comparison information; multichannel input is averaged down; and **no loudness matching**, because a volume difference is also part of what you're comparing. `swap_channels` flips L/R so you can re-listen and rule out your ears' own left/right bias. Ships with an on-node player, and the output can go straight into SaveAudio.

---

### Any Switch

`nodes/any_switch.py` · `ui/any_switch.js` · 分类 Category：`utils/switch`

**中** —— 可视化多路开关：面板里勾选第几路就输出第几路（单选），一个都不选则输出 None。输入是 Any 类型的可增长列表，连满会自动长出下一个输入口（最多 10 路）。`mode` 切到 random 后每次运行随机挑一路输出——随机发生在**前端排队时**，会写进 prompt 从而改变缓存签名，因此本节点和下游每次都真正重跑，这正是随机模式要的语义；后端只在 selection 无效时兜底随机一次。列表里每行的条目名可以**双击原地改名**（回车确认、Esc 取消、留空恢复默认名）：名字存 `labels_json` 隐藏 widget，只随工作流文件持久化、不进 prompt，因此改名不会让下游缓存失效。面板高度随条目数自动增减（恒留 10px 余量），列表不会出现滚动条，多路时也不必手动把节点拉长。选中的那一路和模式都是普通 widget，随工作流文件持久化。**下游需要容忍 None。**

**EN** — A visual multi-way switch: tick the Nth slot in the panel and that slot is output (single-select); tick nothing and you get None. Inputs are an autogrowing Any list that adds a slot as soon as you fill it (up to 10). In `mode=random` each run picks one connected input at random — the draw happens **in the front end at queue time**, so it lands in the prompt, changes the cache signature, and genuinely re-runs this node and everything downstream, which is exactly the semantics random mode needs. The back end only falls back to its own draw when the selection value is unusable (e.g. the API path with no front-end script). Every row's entry name is **double-click renameable** (Enter commits, Esc cancels, empty restores the default): names live in the hidden `labels_json` widget, which is saved with the workflow file but kept out of the prompt, so renaming never invalidates a downstream cache. The panel grows and shrinks with the entry count (always 10px of slack), so the list never scrolls and there is no need to stretch the node by hand. Both the picked slot and the mode are plain widgets, so they persist with the workflow. **Downstream nodes must tolerate None.**

---

### MiniMax H3 AV LoRA Stack

`nodes/h3_av_lora.py` · 分类 Category：`MiniMax-H3/LoRA`

**中** —— 给 MiniMax H3 音频+视频扩散模型一次加载最多 8 路 LoRA，每路除了常规 `strength` 之外还有独立的 `audio_weight` 与 `video_weight`，可以把某路 LoRA 的影响偏向音频 token、偏向视频 token，或者两路等权。`loaded_info` 输出逐路的模块数、被跳过的文件、形状不匹配、不支持的变体等诊断信息（这些情况都按警告处理而非报错）。非平凡行为：所有 LoRA 的音频权重与视频权重都相等时走标准 `add_patches` 合并快路径；一旦出现音视频分权，整套切换为逐模块 forward 注入。输入不是 H3 模型（缺 `diffusion_model.blocks`）会直接报错；`chunks` 需要与 SolAttn 的 "MiniMax H3 Chunk FeedForward" 设置保持一致，否则分块形状对不上，音视频分权会退回不生效并打一次服务端警告。

**EN** — Loads up to 8 LoRAs onto a MiniMax H3 audio+video diffusion model at once. Besides the usual `strength`, every slot carries its own `audio_weight` and `video_weight`, so a LoRA can be biased toward audio tokens, toward video tokens, or kept balanced across both. `loaded_info` returns a per-LoRA diagnostic string: module counts, skipped files, shape mismatches, unsupported variants, final wrap stats — those are reported as warnings rather than raised. Non-obvious behaviour: when every LoRA has identical audio and video weights the node takes the standard merged `add_patches` fast path; as soon as any modality split appears, the whole stack switches to per-module forward injection. A non-H3 input (no `diffusion_model.blocks`) is a hard error. `chunks` must match SolAttn's "MiniMax H3 Chunk FeedForward" setting — mismatched chunk shapes fall back to a plain divisor, AV masking does not apply, and a one-time server warning is logged.

---

### Aspect Ratio Selector

`nodes/aspect_ratio_selector.py` · `ui/aspect_ratio_selector.js` · 分类 Category：`image/aspect_ratio`

**中** —— 可视化分辨率选择器：在节点上拖矩形直接定宽高比，后端按「总像素数」把比例换算成整数 W×H 输出。`total_megapixels` 单位是万像素（100 = 100 万 = 1 MP）；`min_step` 保证输出的宽高都是它的整数倍（默认 8，适配 VAE）。第三个输出 `actual_megapixels` 是取整之后的真实总像素，用来确认对齐步长带来的偏差。

**EN** — A visual resolution picker: drag a rectangle on the node to set the aspect ratio, and the back end converts that ratio plus a total pixel budget into integer W×H outputs. `total_megapixels` is in units of 10k pixels (100 = 1,000,000 = 1 MP); `min_step` forces both edges to be multiples of it (8 by default, for VAE compatibility). The third output, `actual_megapixels`, reports the real budget after snapping so you can see how much the step rounding moved it.

---

### Model Reroute (Step LoRA Switch)

`nodes/model_reroute.py` · 分类 Category：`model/reroute` · 详见 see [`docs/model_reroute.md`](docs/model_reroute.md)

**中** —— 在同一底模的两条 LoRA 分支之间，于采样中途（到达指定步数或 σ 阈值）热切换模型，让**同一次** KSampler 运行的前段用 LoRA A、后段用 LoRA B 去噪。`trigger=percent`（默认）按噪声水平切换，`switch_percent=0.2` 表示前 20% 的步用 A、之后用 B，与 Sol-Attn 的 `start_percent` 同刻度、对步数和采样器不敏感；`trigger=step` 按本次采样的实际调度步数切换。
关键约束与兼容性：两条分支必须来自同一个 checkpoint 对象；切换只替换两分支**差异**的权重补丁（`patches` / `object_patches` / `injections`），不触碰 `transformer_options`，因此与 Sol-Attn、Spectrum 这类基于 model_options 的节点兼容，本节点下游新增的补丁在两侧都保持生效；切换是一次性 A→B，每次采样结束自动恢复 A，同一个 MODEL 输出可复用于多个 KSampler；multi-GPU 并行采样下自动禁用并告警。块级缓存加速器（HyperStep / Teacache / EasyCache）在切换点后有缓存陈旧风险，建议把切换点避开它们的活动窗口。

**EN** — Hot-swaps between two LoRA branches of the same base model *mid-sampling*, so one single KSampler run denoises with LoRA A early and LoRA B late. `trigger=percent` (default) switches on the noise level — `switch_percent=0.2` means A for the first 20% of steps, B after — same scale as Sol-Attn's `start_percent`, insensitive to step count and sampler choice; `trigger=step` switches on the actual scheduled step index.
Constraints and compatibility: both branches must originate from the same checkpoint object. The swap only replaces the weight patches that actually **differ** between the branches (`patches` / `object_patches` / `injections`) and never touches `transformer_options`, so transformer_options-based nodes like Sol-Attn and Spectrum keep working, and patches added downstream of this node stay live on both sides. The switch is one-way A→B and branch A is restored at the end of every run, so the same MODEL output can be reused across multiple KSamplers deterministically. It disables itself with a warning under multi-GPU parallel sampling. Block-level cache accelerators (HyperStep / Teacache / EasyCache) risk stale caches after the switch point — keep the switch out of their active window.

---

### LoRA Step Schedule (Bypass)

`nodes/lora_step_schedule.py` · `ui/lora_step_schedule.js` · 分类 Category：`loaders`

**中** —— 不把 LoRA 合并进底模权重，而是把 LoRA 的 down/up 分支注入每个目标模块的 forward；采样每一步按当前 σ 对应的实际步号更新一次强度，所以同一个 LoRA 可以前段强、后段弱，或者反过来。用 `start_strength` / `end_strength` + 单调曲线（`linear` / `ease_in` / `ease_out` / `ease_in_out` / `cosine`）+ `points` 生成强度列表；`strength_list` 非空时优先使用手填列表（逗号/空格/换行分隔，支持中文逗号）。节点下方是曲线面板：虚线是曲线参考，实线是当前列表，上下拖动圆点会实时把整份列表写回 `strength_list`；“曲线生成”把当前曲线落成显式列表，“清空”恢复自动生成。
步数对齐：按本次采样 `sample_sigmas` 的实际步号取强度；实际步数多于列表时末尾保持最后一个值，少于列表时只取前几个值。`precision` 只控制 LoRA 分支（down/up matmul）的计算精度，默认 `bf16`，可选 `fp16` / `fp32`；底模精度不变。底模权重不被修改，LoRA 分支在模块 forward 里叠加，因此可与权重 offload/量化模型共存。多个本节点**串联在同一条 MODEL 链**上时会自动合并：每个模块只挂一个 hook、只 clone 一次输出，各 LoRA 仍按自己的列表更新强度；并行分支不会合并（它们本来就应该各走各的采样）。日志会打印 `合并 N 个动态 LoRA：M 个模块，K 个适配器，X 个拼接组 / Y 个单独分支`。同一模块/切片上的多路标准 LoRA 会用秩拼接：down/up 预拼一次，强度只在 rank 空间缩放；额外显存约等于被拼接适配器权重的一份拷贝（这个 MOSS LoRA 在 bf16 下约 148MB/路），模型 detach 时释放；LoHa/LoKr/mid 仍单独计算。对 MOSS-Image 这类 fused qkv LoRA，会按 core key map 的 `(key, (0, start, length))` 把 q/k/v 的 delta 加到同一个 qkv 输出的对应切片。挂载时只读模块类型和属性，不访问 `module.weight`，避免 MOSS-Image int8 动态加载器逐层 staging（178 个模块约 10 秒的开销）。
限制：支持标准 LoRA / LoHa / LoKr，以及 fused qkv 输出切片键（MOSS-Image q/k/v、Qwen gate_up、Flux3 linear1_qkv；仅 dim=0 输出切片）；带 function 变换的 tuple 键会明确报错。DoRA（`dora_scale`）、`reshape_weight`、`diff`/`set` 补丁也会明确报错，而不是静默降级。只处理 MODEL，不处理 CLIP。步号依赖采样器的 `predict_noise` 路径；没有 `sample_sigmas` 的自定义采样路径会退化为按模型调用次数递增，完全不经过 `predict_noise` 的路径只会以第一个强度参与。multi-GPU 并行下其余 GPU 分片可能仍按初始强度计算，会打一次警告。

**EN** — Instead of merging LoRA into the base weights, this node injects the LoRA down/up branch into each target module's forward pass and updates the multiplier once per sampling step from the actual sigma. One LoRA can therefore be strong early and weak late, or the reverse. `start_strength` / `end_strength` plus a monotonic curve (`linear` / `ease_in` / `ease_out` / `ease_in_out` / `cosine`) and `points` generate the schedule; when `strength_list` is non-empty it takes priority (comma/space/newline separated, full-width punctuation accepted). The panel below the node draws the reference curve as a dashed line and the active list as a solid line; dragging a point up or down writes the whole list back to `strength_list` in real time. “曲线生成” materializes the current curve into an explicit list, and “清空” returns to automatic generation.
Step alignment uses the actual step index derived from the sampler's `sample_sigmas`; if the run has more steps than the list the last value is held, and if it has fewer steps only the prefix is used. `precision` controls only the LoRA branch (down/up matmul, default `bf16`, optional `fp16` / `fp32`); the base model's dtype is untouched. Base weights are never modified — the branch is added in the forward pass, so it coexists with weight offloading and quantized models. Multiple instances **stacked on the same MODEL chain** are merged automatically: one hook per module and one output clone, while each LoRA keeps its own schedule; parallel branches are not merged (they are meant to sample independently). The log prints `合并 N 个动态 LoRA：M 个模块，K 个适配器，X 个拼接组 / Y 个单独分支`. Standard LoRAs sharing a module/slice use rank concatenation: down/up are concatenated once and strength is applied in rank space; extra VRAM is roughly one copy of the concatenated adapter weights (about 148 MB per LoRA for this MOSS file in bf16), released when the model is detached. LoHa/LoKr/mid adapters stay on the individual path. For fused qkv LoRAs such as MOSS-Image, core key maps emit `(key, (0, start, length))`: the q/k/v deltas are added to the corresponding slices of one qkv output. Attach reads only module type/attributes, never `module.weight`, avoiding the per-layer staging that MOSS-Image's int8 dynamic loader would otherwise perform (~10 s for 178 modules).
Limits: standard LoRA / LoHa / LoKr plus fused-projection output-slice keys (MOSS-Image q/k/v, Qwen gate_up, Flux3 linear1_qkv; dim=0 slices only); tuple keys with a function transform raise a clear error. DoRA (`dora_scale`), `reshape_weight`, and `diff`/`set` patches also raise a clear error instead of silently degrading. MODEL only, no CLIP. Step tracking relies on the sampler's `predict_noise` path; custom samplers without `sample_sigmas` fall back to a model-call counter, and paths that never call `predict_noise` only use the first strength. Under multi-GPU parallel sampling the auxiliary GPU shards may keep the initial strength; a one-time warning is logged.

---

### Mask Paint Output

`nodes/mask_paint_output.py` · `ui/mask_paint_output.js` · 分类 Category：`image/mask`

**中** —— 预览上游图片、在其上用**官方遮罩编辑器**绘制遮罩，然后成对输出 IMAGE 与 MASK。右键节点选「用遮罩编辑器打开」（或图片工具栏的遮罩按钮）唤出绘制工具，也支持直接粘贴/拖入底图。IMAGE 与 MASK 永远取自同一张绘制过的 PNG（RGB → IMAGE，alpha → MASK），所以节点上显示的和下游拿到的必定是同一对，不会出现"看着有遮罩、输出是空的"。
`stop_here` 是模式开关：开启 = 取图模式，本节点静默阻断下游（不报错、不弹窗）只用来画图，从关闭切回开启、或上游换了新图，都会作废当前图与遮罩改用上游图；关闭 = 输出模式，当前图与遮罩被锁定，上游怎么变化都不影响本节点，下游正常执行。两者分辨率对不上时 MASK 退回全零——错位的遮罩比空遮罩更危险，而空遮罩在预览里一眼可见。

**EN** — Previews an upstream image, lets you paint a mask on it with ComfyUI's **official mask editor**, and outputs the IMAGE/MASK pair. Open the editor from the node's context menu ("Open in MaskEditor") or the image toolbar; pasting or dropping a base image works too. IMAGE and MASK always come from the *same* painted PNG (RGB → IMAGE, alpha → MASK), so what the node shows and what downstream receives are guaranteed to be the same pair — no "the mask looked drawn but the output is empty".
`stop_here` is the mode switch: on = capture mode, this node silently blocks downstream (no error, no popup) so you can just paint; flipping from off back to on, or a genuinely new upstream image, discards the current image and mask and adopts the upstream one. off = output mode, the current image and mask are locked, further upstream changes no longer affect this node, and downstream runs normally. If the two resolutions don't match, MASK falls back to all-zeros — a misaligned mask is more dangerous than an empty one, and an empty one is obvious in the preview.

---

### Persistent Output

`nodes/persistent_output.py` · `ui/persistent_output.js` · 分类 Category：`utils/compare`

**中** —— 在 3 个独立槽位里跨运行记住输入（每个槽位 = 一批图 + 一段音频）。典型流程：第一趟把上游连到槽 1，第二趟连槽 2，第三趟连槽 3，然后**断开上游**——三个槽位各自复现它记住的内容，A/B/C 并排对比。
每个槽位三种状态：连着输入 → 存储被替换，输出是上游的**原始对象**（透传）；没连输入但有存档 → 输出存档的一份**新建临时副本**（复现）；从没存过 → 输出 None，因此**下游必须容忍 None**。存档只存在内存里、按节点 unique_id 索引，进程退出即消失；删除节点会顺带清掉存档，槽位上也有逐槽释放按钮。存档刻意用 numpy 缓冲而不是 torch 张量，因为在本进程里 torch 分配过的块基本不归还已提交内存——代价是桥接/复现时有一段双份内存，这是有意取舍。`gen` 是内部计数器，不要手改。

**EN** — Remembers its inputs across prompt runs in 3 independent slots, each holding an image batch plus an audio pair. Typical flow: wire upstream into slot 1 on run 1, slot 2 on run 2, slot 3 on run 3, then **disconnect the upstream** — each slot replays what it kept, giving you A/B/C side by side.
Per slot there are three states: connected → the store is replaced and the output is the **original upstream object** (bridge); not connected but stored → the output is a **fresh temporary clone** of the stored copy (replay); never stored → output is None, so **downstream must tolerate None**. Storage lives in RAM only, is keyed by the node's unique_id, and is gone at process exit; deleting the node clears it, and each slot has its own release button. The private copy is deliberately backed by numpy buffers rather than torch tensors, because torch-allocated blocks in this process essentially never give committed memory back — the price is a window of doubled RAM during bridge/replay, which is the intentional trade-off. `gen` is an internal counter; don't touch it.

---

### Refreshable Text Box

`nodes/refreshable_text_box.py` · `ui/refreshable_text_box.js` · 分类 Category：`text`

**中** —— 显示会自刷新的文本，且刷新不污染运行缓存。痛点是：文本框节点每次运行后把上游文本写回自己的 widget，下一次 prompt 里这个字面量就变了，运行缓存从本节点起整片失效——即使上游文本根本没变。而后端没有任何"某输入不参与缓存"的机制，`IS_CHANGED` 也救不了（变化的字面量照样进签名，它只能增加失效、不能减少）。所以由前端给文本 widget 挂 `serializeValue`：未锁定时一律发送空串常量 → prompt 字节稳定 → 本节点与下游缓存不失效；锁定后才发送真实文本（此时失效是预期的，你就是要让改过的文本生效）。
典型用法：开着「允许覆盖」→ 运行到此节点 → 在文本框里编辑 → 关掉允许覆盖（锁定）→ 再运行。文本照样会随工作流文件持久化，不必为了不失效而牺牲保存。

**EN** — Shows text that refreshes itself without letting that refresh bust the run cache. The pain it removes: a text-box node that writes upstream text back into its own widget changes that literal in the next prompt, so everything downstream re-runs even when the upstream text never changed. There is no back-end mechanism for "this input shouldn't affect caching", and `IS_CHANGED` can't help either — a changed literal still enters the signature, and it can only add invalidations, never remove them. So the fix lives in the front end: the text widget's `serializeValue` sends a constant empty string while unlocked, keeping the prompt byte-stable and the cache intact; only after locking does it send the real text (and then the invalidation is expected — that's the point of editing).
Typical loop: leave "allow overwrite" on → run up to this node → edit the text → unlock (switch overwrite off) → run again. The text still persists with the workflow file, so you don't have to give up saving to avoid the invalidation.

---

### CLIP Text Encode (Cached)

`nodes/cached_clip_text_encode.py` · 分类 Category：`conditioning`

**中** —— 与核心 CLIP Text Encode 等价，但按 (CLIP 对象, 文本) 缓存编码结果。痛点与 Refreshable Text Box 同源：ComfyUI 的缓存签名会把输入字面量和祖先签名一起纳入，所以 LoRA 开关、元数据同步、文本框刷新都会让下游文本编码重跑一次——哪怕最终文本一字未变。对 Qwen3.5-9B 这类大文本编码器，这次重编码就是数秒。本节点在 encode 外面加一层 4 条的小 LRU：命中时直接返回上次的 conditioning，不重新 tokenize/encode；文本或 CLIP 真正变化时照常重编码。带 hooks 的 conditioning 不缓存，`cache_mode=disabled` 可临时绕过。它只替换显式使用本节点的编码，不做全局 monkey-patch。
用法：把工作流里核心的 `CLIP Text Encode` 直接换成这个节点，`clip` / `text` 照原样接上，`cache_mode` 保持 `enabled`。缓存只持有 conditioning，CLIP 模型本身用 `weakref` 引用，不会阻止模型卸载或换出。缓存键还包含 `patcher.patches_uuid`：标准 `add_patches` 原地改权重也会换 uuid，因此同一对象、patch 数量不变的原地 LoRA 也能正确失效；只有绕过 `add_patches`、直接改底层权重（如 `load_state_dict`）、直接改 `patcher.patches`，或替换 `cond_stage_model` / `patcher` 的极端情况才需要 `cache_mode=disabled`。
注意：节点本身仍会被 ComfyUI 执行（上游签名变了），只是执行成本降到一次字典查找；若要让节点完全不执行，需要把触发词来源与 loader 解耦（独立节点 + 前端同步），不在本节点能力范围内。

**EN** — Drop-in equivalent of the core CLIP Text Encode that caches the encoded conditioning by (CLIP object, text). Same root cause as Refreshable Text Box: ComfyUI's cache signature includes input literals plus ancestor signatures, so toggling a LoRA, syncing metadata, or refreshing a text box re-runs the downstream text encode even when the final text is byte-identical. With a Qwen3.5-9B text encoder that costs seconds per run. This node wraps `encode` in a 4-entry LRU: on a hit it returns the previous conditioning without tokenizing/encoding again; a real text or CLIP change still re-encodes. Conditioning that carries hooks is never cached, and `cache_mode=disabled` bypasses the cache. It only affects encodes that explicitly use this node — no global monkey-patching.
Usage: replace a core CLIP Text Encode with this node, wire `clip` / `text` as before, and leave `cache_mode` on `enabled`. The cache holds conditioning only; the CLIP model is referenced through a `weakref`, so it won't block unloading or swapping. The cache key also includes `patcher.patches_uuid`: a standard `add_patches` call changes that uuid even when it patches the same object in place, so an in-place LoRA with an unchanged patch count still invalidates correctly. Only extreme cases that bypass `add_patches` — direct base-weight edits such as `load_state_dict`, direct edits to `patcher.patches`, or replacing `cond_stage_model` / `patcher` — need `cache_mode=disabled`.
Caveat: the node itself is still executed by ComfyUI (its ancestor signature changed); the cost drops to a dict lookup. Making the node not execute at all requires decoupling the trigger-word source from the loader (a separate node plus front-end syncing), which is out of scope here.

---

### Image Container (6 Outputs)

`nodes/image_container.py` · `ui/image_container.js` · 分类 Category：`image/container`

**中** —— 多图容器：往节点里攒一批图，再点选决定哪几张输出、按什么顺序输出。三种进图方式：点按钮上传、把图片文件拖到节点上、Ctrl+V 粘贴剪贴板图片。单击缩略图选中它，按点击先后占用输出槽 1..6，左上角角标显示它占第几个槽，再点一次取消、后面的角标自动进位；长按拖拽只调整容器内的显示顺序，不影响角标编号；缩略图右上角 × 把该图移出容器。输出是 6 个单图槽 + 1 个 batch 槽（把选中的图合成一个批次，尺寸不一致时跳过合成，单图槽不受影响）。
`unselected_output` 决定没被选中的槽产出什么，差别在下游怎么反应：`none`（默认）该槽输出 None，下游照常运行只是少一张，适合多条线汇入同一个多图输入节点（前提是该入参可选且会判 None）；`block` 静默阻断连到它的节点且继续往更下游传染，适合每个槽各喂一条独立分支、没选就不想跑那条分支，注意多个槽汇入同一个节点时只要有一个没选那个节点就整个不跑；`placeholder` 输出一张与第一张选中图同尺寸的纯黑图，下游一定会跑但会把黑图当真图处理，只在下游既不吃 None 也不吃阻断时才用。
工具条「全不选」左侧有三个点选模式按钮：**多选**（默认，即上文的点选累加行为）、**单选**（同时只允许选中一张；从多选切过来时自动收拢为容器显示顺序的第一张；点已选中的那张保持不变，点另一张整体替换；可用「全不选」清空）、**顺序单选**（单选的一切行为 + 排队即推进：每次排队时本次运行仍用当前选中的图，选中项随即推进到容器里的下一张，连点多次排队会依次用第 1、2、3…张，到最后一张就停住不再动）。模式选择随工作流文件保存、不进 prompt（不影响缓存签名）。顺序单选的推进挂在前端 `serializeValue` 上（每次排队构建 prompt 时调用）：推进发生在排队瞬间，与运行成败无关——失败、取消、缓存命中都已推进；也正因如此，顺序模式下 selected_json 每次排队都不同，本节点每跑必重跑、下游跟着重跑，这是逐张步进的内在代价。

**EN** — A multi-image container: collect a pool of images on the node, then click to decide which ones get output and in what order. Three ways in: the upload button, dragging image files onto the node, or Ctrl+V from the clipboard. A single click selects a thumbnail and it takes the next free output slot 1..6 in click order, with its slot number shown as a corner badge; click again to drop it and the badges renumber; long-press-drag only reorders what's displayed inside the container and doesn't touch badge numbers; the × on a thumbnail removes it from the pool. Outputs are 6 single-image slots plus one batch slot that concatenates the selected images — skipped when sizes don't match, which leaves the single-image slots untouched.
`unselected_output` picks what an unselected slot yields, and the three options differ in how downstream reacts: `none` (default) outputs None so downstream still runs with one fewer image — good when several lines converge on the same multi-image node, provided that input is optional and None-checked; `block` returns an ExecutionBlocker, silently skipping the node it feeds (no error, no popup) and propagating further downstream — good when each slot drives its own branch and an unselected branch simply shouldn't run, but note that if several slots converge on one node, any single unselected slot stops that whole node; `placeholder` emits a black image matching the first selected image's size, so downstream definitely runs but treats black frames as real content with no warning — use it only when downstream tolerates neither None nor blocking.
To the left of the "deselect all" button the toolbar has three selection-mode buttons: **multi** (default, the click-to-accumulate behaviour described above), **single** (only one image can be selected at a time; switching over from multi collapses the selection to the first image in container order; clicking the selected image keeps it, clicking another one replaces it; "deselect all" still clears), and **sequential** (everything single does, plus queue-time stepping: when a prompt is queued this run still uses the currently selected image, and the selection then advances to the next image in container order — queueing N times back-to-back uses images 1..N in order, and it stays on the last one once reached). The mode is saved with the workflow file and never enters the prompt (no cache-signature impact). Sequential stepping is hooked into the front end's `serializeValue` (called each time a prompt is built on queue): the advance happens at queue time and cannot know the run's outcome — failures, cancels and cache hits advance all the same; for the same reason selected_json differs on every queue in sequential mode, so this node always re-runs and drags its downstream with it — an inherent cost of per-run stepping.

---

### Video Combine (Memory)

`nodes/psl_video_memory.py` · 分类 Category：`video/psl`

**中** —— 帧 + 可选音频直接在内存中合成 MP4，全程不落盘：ffmpeg 以 fragmented MP4 写 stdout，Python 侧用 BytesIO 接住后包成官方 VIDEO 类型（`VideoFromFile(BytesIO)`），可无缝接官方一切 VIDEO 下游。编码引擎复刻本机优化过的 VHS VideoCombine，三项加速机制全保留：① GPU 往返量化——CPU 大帧搬上 GPU 做乘/clamp/量化，只把小体积 uint8 传回，转换耗时约为纯 CPU 的 1/3 且显存峰值恒定；② 有界队列流水线——帧转换在 daemon 线程经 maxsize=4 的队列与 NVENC 编码重叠，队列满自动背压；③ NVENC 硬编 + fake_trc 色彩参数——输入按 sRGB 谎报为 BT.709 传递函数，输出侧 `scale=out_color_matrix=bt709` 消除 gamma 漂移。默认参数 `nvenc_h264-mp4 / yuv420p / 15 Mbps`，另有 `nvenc_hevc-mp4` 与 libx264 软编（crf/preset 可调）备选。与 VHS 的结构差异除输出内存化外还有音频：VHS 先落盘无声视频再用第二个 ffmpeg 进程 mux，本节点把音频写成临时 f32le 裸流文件与视频在**同一条命令**里完成编码合成；顺带修掉 VHS 流水线"转换线程异常被静默、产出残缺文件"的隐患（异常经队列回传主线程显式报错）。多段合并接官方 ConcatenateVideo、落盘接官方 SaveVideo——编码参数一致时都是纯 remux 零重编码。注意两点：① 需要系统 PATH 上有 ffmpeg（full build）；② 中间视频驻留内存（15 Mbps 约 112 MB/分钟/段，输出缓存会持有引用），长会话多段拼接注意内存余量。

**EN** — Turns frames plus optional audio into an MP4 entirely in memory, never touching disk: ffmpeg writes a fragmented MP4 to stdout, Python collects it into a BytesIO and wraps it as the official VIDEO type (`VideoFromFile(BytesIO)`), so it plugs into any official VIDEO downstream. The encoding engine replicates the locally-tuned VHS VideoCombine, keeping all three speedups: (1) GPU round-trip quantization — large CPU-resident frames are moved onto the GPU for scale/clamp/cast and only the small uint8 result travels back, cutting conversion time to about a third with a flat VRAM ceiling; (2) a bounded-queue pipeline — frame conversion runs on a daemon thread and overlaps NVENC encoding through a maxsize=4 queue with natural back-pressure; (3) NVENC hardware encoding with fake_trc color args — input is labelled BT.709 (a benign lie for sRGB data) and the output side uses `scale=out_color_matrix=bt709` to kill the gamma drift. Defaults are `nvenc_h264-mp4 / yuv420p / 15 Mbps`, with `nvenc_hevc-mp4` and libx264 software encoding (crf/preset) as alternates. Two structural differences from VHS besides the in-memory output: audio — VHS writes a silent video first and muxes with a second ffmpeg process, while this node feeds audio as a temporary raw f32le file so one single command encodes and muxes everything; and a fixed hazard — VHS's conversion thread fails silently on error, producing truncated files, while here the exception travels back through the queue and fails loudly. For multi-segment merging wire it into the official ConcatenateVideo and SaveVideo — with matching encode parameters both are pure remux with zero re-encoding. Two caveats: `ffmpeg` must be on the system PATH (full build), and the intermediate video lives in RAM (about 112 MB per minute per segment at 15 Mbps; the output cache holds a reference), so mind memory on long sessions with many segments.

---

### Advanced Video Match V2 (PSL)

`nodes/video_match_v2.py` · 分类 Category：`video/psl`

**中** —— 把上一段视频的尾部色调对齐到下一段视频的开头，替代手工调色。移植自 SVI_with_PainterI2V 的 `SVI_AdvancedVideoMatchV2`。参考侧取尾部 `ref_anchor_frames` 帧做抗崩坏加权锚点；目标侧默认 `dual_anchor`：首帧完全对齐参考基准，随后按余弦曲线平滑过渡回目标视频自身的尾段基准，避免帧间统计跳变。匹配在 Oklab 空间做逐通道均值/方差映射，可选叠加 CDF 直方图修正（`affine_cdf`），另有 `affine_only` / `cdf_only`；`cdf_max_shift` 与 `cdf_chroma_guard` 防止饱和色被推得过深过艳。`frozen` 复用首帧的固定变换，`rolling` 保留旧版逐帧 EMA 行为。
注意 `dual_anchor` 会逐帧用当前帧的 mean/std 归一化，目标片段自身的曝光变化会被压平，淡出结束时可能看到明度回落；有曝光变化/淡出/溶解时推荐 `stable_anchor`：自动从全片段筛稳定帧（排除低对比、跳变、离群帧），参考侧取最近稳定窗口、目标侧用稳定帧基准，再用一条固定 B→A 映射按强度曲线混合，保留目标内部的曝光/对比动态；同时自动限制整帧平均明度的净偏移，并按当前帧对比度降低溶解/闪白帧的修正强度。`cross_channel` 在 `stable_anchor` / `frozen` / `rolling` 下生效，`dual_anchor` 是逐通道映射，该参数无效。
输出是「参考帧 + 修正后的目标帧」拼接，可直接替换原目标片段做对比。默认参数来自实际工作流：`window_size=3 / cdf_weight=0.50 / cross_channel=0.10 / cdf_max_shift=0.08 / cdf_chroma_guard=0.90`。

**EN** — Matches the tail color of the previous clip to the head of the next clip, replacing manual color grading. Ported from `SVI_AdvancedVideoMatchV2` in SVI_with_PainterI2V. The reference side builds a robust anchor from the last `ref_anchor_frames` frames (consistency-weighted against a single corrupted frame). The target side defaults to `dual_anchor`: frames start fully aligned to the reference baseline and smoothly interpolate back to the target clip's own tail baseline along a cosine curve, avoiding frame-to-frame statistic jumps. Matching runs in Oklab with per-channel mean/std plus optional CDF histogram correction (`affine_cdf`), with `affine_only` / `cdf_only` alternatives; `cdf_max_shift` and `cdf_chroma_guard` keep saturated colors from being pushed too deep or too vivid. `frozen` reuses a single transform computed from frame 0, and `rolling` preserves the old per-frame EMA behavior.
Note that `dual_anchor` normalizes every frame with its own mean/std, flattening the target clip's own exposure changes and potentially producing a brightness drop when the fade ends. For clips with exposure changes, fades or dissolves, prefer `stable_anchor`: it automatically filters stable frames across the whole clip (dropping low-contrast, jumpy and outlier frames), anchors the reference on the latest stable window and the target on its stable-frame baseline, then blends one fixed B→A transform along the strength curve — preserving the target's internal exposure/contrast dynamics. It also caps the frame-mean luminance shift and reduces correction strength on dissolve/flash frames according to their contrast. `cross_channel` is active in `stable_anchor` / `frozen` / `rolling`; `dual_anchor` is a per-channel mapping where that parameter has no effect.
The output concatenates the reference frames with the corrected target frames, so it drops straight into a comparison or the next stage. Defaults come from a tuned workflow: `window_size=3 / cdf_weight=0.50 / cross_channel=0.10 / cdf_max_shift=0.08 / cdf_chroma_guard=0.90`.

---

## 仓库结构 / Layout

```
__init__.py                 注册入口 WEB_DIRECTORY=./ui · registration entry, WEB_DIRECTORY=./ui
nodes/                      节点后端 node back ends
  audio_filter/             Harshness Suppressor 及其 DSP 实现与脱离服务端可跑的测试
                            suppressor + its DSP core and tests that run without a server
ui/                         各节点的前端脚本 front-end scripts, served under /extensions/<pack name>/
tests/                      脱离服务端可跑的节点单测 node unit tests that run without a server
docs/model_reroute.md       Model Reroute 的用法、限制与兼容性表
                            usage, limits and compatibility matrix for Model Reroute
```

## 备注 / Notes

- 本包是个人工作流里长出来的工具集合，节点之间没有依赖关系，可以只用其中任意几个。
  These nodes grew out of one person's workflows. They don't depend on each other — use any subset.
- `audio_filter/dsp.py` 不依赖 ComfyUI，可直接 `python nodes/audio_filter/tests/test_dsp.py` 跑测试。
  `audio_filter/dsp.py` is ComfyUI-free; run `python nodes/audio_filter/tests/test_dsp.py` to test it standalone.
- 多个节点会**主动输出 None 或阻断下游**（Any Switch / Image Container / Persistent Output / Mask Paint Output），这是设计而非 bug，相关小节各自说明了取舍。
  Several nodes deliberately output None or block downstream (Any Switch / Image Container / Persistent Output / Mask Paint Output). That's design, not a bug — each section explains the trade-off.

## License

未附许可证文件 / No license file is included in this repository.
