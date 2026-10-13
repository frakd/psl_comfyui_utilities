"""Audio A/B Compare — 两路音频合成一条左右分声道的对比音频。

左声道播放 A、右声道播放 B：同一副耳机就能同时听两路内容做 A/B 对比，
不必来回切换播放。核心已有的 JoinAudioChannels 只收 mono、把长的一边裁短、
也不带预览，对「对比」这个场景刚好三处都不合适，所以本节点自己实现：

- 采样率不同：只把较慢的一路**上采样**到较快的一路（comfy.audio.resample，
  带限 sinc，与核心 VAEEncodeAudio 同源）。只升不降，不丢信息——对比节点
  任何一塊有损处理都会污染对比结论。
- 长度不同：短的一路**补静音**到较长一边，不裁剪。哪边先结束在听感上
  一目了然，这本身就是对比信息。
- 多声道输入：按声道取平均下混成单声道再进耳朵。对比要的是每只耳朵
  一路独立信号，立体声声像没有意义。
- 不做响度匹配：两路的音量差也是对比对象，自动增益会把它抹掉。
"""

import torch

import comfy.audio
from comfy_api.latest import io, UI


class AudioABCompare(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioABCompare",
            search_aliases=["ab", "abx", "compare", "对比", "左右耳", "双耳", "声道"],
            display_name="Audio A/B Compare",
            category="audio/compare",
            description=(
                "把两路音频合并成一条左右分声道的对比音频：A 进左耳、B 进右耳，"
                "戴耳机即可同时听两路做 A/B 对比。采样率不同时较慢的一路被上采样到"
                "较快的一路；长度不同时短的一路补静音（不裁剪）；多声道输入下混为"
                "单声道。不做响度匹配——音量差也是对比对象。节点自带预览可直接播放，"
                "输出也可接 SaveAudio 保存成文件。"
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input(
                    "audio_a",
                    display_name="音频 A（左耳）",
                    tooltip="进左声道的一路。多声道会下混成单声道。",
                ),
                io.Audio.Input(
                    "audio_b",
                    display_name="音频 B（右耳）",
                    tooltip="进右声道的一路。多声道会下混成单声道。",
                ),
                io.Boolean.Input(
                    "swap_channels",
                    display_name="左右互换",
                    default=False,
                    tooltip=(
                        "开启后 A 进右耳、B 进左耳。人耳对左右灵敏度有差异，"
                        "换边复听可以排除「某路听起来差是因为耳朵」的错觉。"
                    ),
                ),
            ],
            outputs=[io.Audio.Output(display_name="对比音频")],
        )

    @classmethod
    def _to_mono(cls, waveform: torch.Tensor) -> torch.Tensor:
        """[B, C, N] -> [B, 1, N]：多声道按平均下混，mono 原样返回。"""
        if waveform.shape[1] == 1:
            return waveform
        return waveform.mean(dim=1, keepdim=True)

    @classmethod
    def execute(cls, audio_a, audio_b, swap_channels=False):
        if audio_a is None or audio_b is None:
            raise ValueError(
                "AudioABCompare 需要 A、B 两路音频都有输入；"
                "只想听单路请直接用核心的 Preview Audio。"
            )

        wave_a, rate_a = audio_a["waveform"], audio_a["sample_rate"]
        wave_b, rate_b = audio_b["waveform"], audio_b["sample_rate"]

        # 采样率对齐：目标取两者较高者，只做上采样（升采样不丢信息，降采样会丢）。
        # 一个输出文件左右声道必须同采样率，这里没有无损的第三条路。
        target_rate = max(rate_a, rate_b)
        if rate_a != target_rate:
            wave_a = comfy.audio.resample(wave_a, rate_a, target_rate)
        if rate_b != target_rate:
            wave_b = comfy.audio.resample(wave_b, rate_b, target_rate)

        left = cls._to_mono(wave_a)
        right = cls._to_mono(wave_b)

        # 批次数不同时按核心惯例广播 batch=1 的一侧；两个都大于 1 且不等才是真错误。
        if left.shape[0] != right.shape[0]:
            if left.shape[0] == 1:
                left = left.expand(right.shape[0], -1, -1)
            elif right.shape[0] == 1:
                right = right.expand(left.shape[0], -1, -1)
            else:
                raise ValueError(
                    f"两路音频的 batch 数不一致且无法广播：A={left.shape[0]}, B={right.shape[0]}"
                )

        # 长度对齐：短的一路尾部补静音。不裁剪——对比时哪边先结束本身就是信息。
        target_len = max(left.shape[-1], right.shape[-1])
        if left.shape[-1] < target_len:
            left = torch.nn.functional.pad(left, (0, target_len - left.shape[-1]))
        if right.shape[-1] < target_len:
            right = torch.nn.functional.pad(right, (0, target_len - right.shape[-1]))

        if swap_channels:
            left, right = right, left

        out = {"waveform": torch.cat([left, right], dim=1), "sample_rate": target_rate}
        # 官方预览助手：存 temp flac 并回传 ui，节点上直接出现播放器（同核心 PreviewAudio）。
        return io.NodeOutput(out, ui=UI.PreviewAudio(out, cls=cls))
