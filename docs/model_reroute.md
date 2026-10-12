# Model Reroute (Step LoRA Switch)

在同一底模的两条 LoRA 分支间，于采样中途（到达指定步数 / σ 阈值）热切换模型，使同一次
KSampler 运行的前段用 LoRA A、后段用 LoRA B 去噪。切换只替换两条分支**差异**的
权重补丁（`patches` / `object_patches` / `injections`），不触碰 `transformer_options`，
因此与 Sol-Attn、Spectrum 等基于 model_options 的节点兼容，可任意串联。

## 用法

```
CheckpointLoader ──┬─ LoraLoader(A) ──┬─ model_a ─┐
                   └─ LoraLoader(B) ──┴─ model_b ─┴─ Model Reroute ── KSampler
```

- `model_a` / `model_b`：同一底模（同一 checkpoint 对象）加载的不同 LoRA 组合。
  支持原生 LoraLoader、LoraLoaderModelOnly、H3 AV LoRA Stack（merge 与不等权重
  模式均可）等一切通过 `add_patches` / `set_injections` / object patch 生效的加载器。
- `trigger`：
  - `percent`（默认）：按噪声水平切换，`switch_percent=0.2` 表示前 20% 采样步用 A，
    之后用 B（与 Sol-Attn 的 `start_percent` 同语义、同刻度）。对步数/采样器不敏感。
  - `step`：按本次采样的实际调度步数切换，`switch_step=N` 表示第 N 步起用 B。

## 限制

- 两分支必须共享同一底模对象（`model_a` / `model_b` 应来自同一个 checkpoint 加载器）。
- 切换是一次性的 A→B；每次采样结束自动恢复 A，同一 MODEL 输出可复用于多个 KSampler。
- 仅权重级补丁按分支切换；`transformer_options` 级补丁（attention override、
  `patches_replace`、wrapper、model_sampling 配置如 H3 sigma shift）对整个采样全程生效。
- multi-GPU 并行采样（`multigpu_clones`）下自动禁用切换并告警。
- 普通 ModelPatcher 与动态加载（`--fast`）的 ModelPatcherDynamic 均支持；动态 patcher
  走其受管理的 `unpatch → patch(load_weights=False) → load(dirty=True)` 序列。
- 每次完整的采样 pass 独立触发切换（含 Spectrum 离线回放的第二趟采样），保证两趟轨迹一致。

## 兼容性

| 节点 | 状态 | 说明 |
|---|---|---|
| Sol-Attn | 兼容 | 其 `optimized_attention_override` 与 object 补丁不在分支差集内，切换不受影响 |
| Spectrum | 兼容 | 本节点只加 OUTER_SAMPLE/PREDICT_NOISE wrapper，不触发其 ER-SDE/rollback 的单 wrapper 校验；离线回放双 pass 各自独立切换 |
| HyperStep / Teacache / EasyCache | 注意 | 块级缓存加速器在切换点后存在缓存陈旧风险（质量退化、不崩溃）；建议切换点避开其活动窗口 |
