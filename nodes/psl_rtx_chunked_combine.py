"""PSL RTX 分片放大视频合成节点。

PSLRtxChunkedVideoCombine（RTX Chunked Upscale Video Combine (Memory)）：
    IMAGE batch + 可选整段音频 → 按帧数分片 → 每片用 RTX Video Super
    Resolution 的等比模式（scale by multiplier）放大 → 顺序写入同一条
    内存 fragmented MP4 → 输出官方 VIDEO。视频帧全程不落盘。

与「RTXVideoSuperResolution → PSLVideoCombineMemory」两节点串联的区别：
  1. 同一次 execute 内完成，VSR 引擎只 load 一次并跨分片复用，避免每个
     分片重复初始化；
  2. 每片写完立即释放该片的 GPU 中转张量与放大帧引用，并 empty_cache，
     长视频的瞬时内存只与 frames_per_chunk 相关；
  3. 所有分片写入同一个 ffmpeg 进程（同一编码会话、同一条 stdin），输出
     天然连续，不需要把分段 MP4 再拼接（无时间戳/extradata 兼容问题）。

已知限制：上游 IMAGE batch 本体由 ComfyUI 缓存持有，节点内无法释放；
本节点能省的是放大结果、编码缓冲与显存。音频与 PSLVideoCombineMemory
相同：ffmpeg 只有一条 stdin 被视频帧占用，整段音频写成临时 f32le 裸流
文件后与视频同一条命令 mux（apad + shortest 对齐），该文件在同一次
execute 内删除；视频本身全程走内存。
"""

import gc
import logging
import os
import subprocess
import threading
import time
from io import BytesIO

import torch

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, InputImpl

from .psl_video_memory import (
    ENCODE_ARGS,
    _NVENC_FORMATS,
    _PRESET_CHOICES,
    _build_input_args,
    _build_output_args,
    _terminate,
    _write_audio_raw,
)

try:
    import nvvfx

    _VSR_QUALITY_OPTIONS = list(nvvfx.QualityLevel.__members__)
    _VSR_IMPORT_ERROR = None
except ImportError as exc:  # 未装 nvvfx 时本包其余节点仍可正常加载
    nvvfx = None
    _VSR_QUALITY_OPTIONS = ["ULTRA"]
    _VSR_IMPORT_ERROR = str(exc)

logger = logging.getLogger("psl.rtx_chunked")

# 与 RTXVideoSuperResolution 一致：单次 H2D 中转的像素预算（按输出分辨率计）。
_MAX_BATCH_PIXELS = 1024 * 1024 * 16

_ENCODING_RGB8 = "8-bit RGB"
_ENCODING_RGB10A2 = "10-bit RGB (RGB10A2)"
_IMAGE_ENCODINGS = (_ENCODING_RGB8, _ENCODING_RGB10A2)


def _pack_rgb10a2(image: torch.Tensor) -> torch.Tensor:
    rgb = image.clamp(0.0, 1.0).mul(1023.0).round().to(torch.int32)
    packed = rgb[..., 0] | (rgb[..., 1] << 10) | (rgb[..., 2] << 20) | -1073741824
    return packed.view(torch.uint32).contiguous()


def _unpack_rgb10a2(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(torch.int32)
    red = packed.bitwise_and(0x3FF)
    green = packed.bitwise_right_shift(10).bitwise_and(0x3FF)
    blue = packed.bitwise_right_shift(20).bitwise_and(0x3FF)
    return torch.stack((red, green, blue), dim=-1).to(torch.float32).div_(1023.0)


def _effect_device() -> tuple[torch.device, int]:
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError("RTX Video Super Resolution 需要 CUDA 设备，当前设备为 %s" % device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return device, device_index


def _drain_pipe(pipe, sink: BytesIO) -> None:
    try:
        while True:
            chunk = pipe.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
    except Exception:
        pass  # 失败判定以 returncode + stderr 为准，这里不吞主错误


def _start_encoder(args: list):
    """启动 ffmpeg 子进程并挂 stdout/stderr 排水线程（防止管道写满卡死）。"""
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_buf = BytesIO()
    err_buf = BytesIO()
    reader_out = threading.Thread(target=_drain_pipe, args=(proc.stdout, out_buf), name="psl-rtx-ffmpeg-stdout", daemon=True)
    reader_err = threading.Thread(target=_drain_pipe, args=(proc.stderr, err_buf), name="psl-rtx-ffmpeg-stderr", daemon=True)
    reader_out.start()
    reader_err.start()
    return proc, (reader_out, reader_err), out_buf, err_buf


def _finish_encoder(proc, readers, out_buf: BytesIO, err_buf: BytesIO, args: list) -> BytesIO:
    for reader in readers:
        reader.join(timeout=30)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        _terminate(proc, *readers)
        raise RuntimeError(f"ffmpeg 编码超时未退出，参数：{' '.join(args)}") from exc
    stderr_tail = err_buf.getvalue().decode(*ENCODE_ARGS)
    if proc.returncode != 0 or out_buf.tell() == 0:
        raise RuntimeError(
            f"ffmpeg 编码失败 (exit={proc.returncode})，参数：{' '.join(args)}\n{stderr_tail}"
        )
    if stderr_tail:
        logger.warning("ffmpeg 输出了一些诊断信息：\n%s", stderr_tail)
    out_buf.seek(0)
    return out_buf


def _effect_frame_bytes(effect, frame: torch.Tensor, image_encoding: str) -> bytes:
    """单帧 VSR 输出 → rgb24 字节：放大与量化都在 GPU，只回传 uint8。"""
    result = effect.run(frame).image
    output = torch.from_dlpack(result).clone()
    if image_encoding == _ENCODING_RGB10A2:
        output = _unpack_rgb10a2(output)
    else:
        output = output.movedim(0, -1)
    quantized = (output * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8)
    return quantized.cpu().numpy().tobytes()


class PSLRtxChunkedVideoCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PSLRtxChunkedVideoCombine",
            display_name="RTX Chunked Upscale Video Combine (Memory)",
            category="video/psl",
            description=(
                "图片 batch 按帧数分片，每片用 RTX Video Super Resolution 等比放大"
                "（scale by multiplier），逐片写入同一条内存 MP4；全部分片完成后"
                "可选封装整段音轨，输出官方 VIDEO。每片结束立即释放该片放大缓存"
                "与显存，适合长视频低内存放大。"
            ),
            inputs=[
                io.Image.Input("images", tooltip="视频帧 [N,H,W,C]，float32 [0,1]。4 通道时 alpha 会被丢弃（H.264 不支持透明）。"),
                io.Audio.Input("audio", optional=True, tooltip="可选整段音轨：全部分片合成后统一编码为 AAC 封装（apad + shortest 对齐）。"),
                io.Float.Input("fps", default=30.0, min=1.0, max=240.0, step=0.01, tooltip="输入/输出帧率。"),
                io.Int.Input("frames_per_chunk", default=16, min=1, max=4096, step=1, tooltip="每片帧数：越小瞬时内存/显存越低；每片处理完即释放该片放大缓存。"),
                io.Float.Input("scale", default=1.5, min=1.0, max=4.0, step=0.01, tooltip="等比放大倍数（scale by multiplier）；DENOISE/DEBLUR 质量档保持原分辨率、忽略此值。"),
                io.Combo.Input(
                    "quality",
                    options=_VSR_QUALITY_OPTIONS,
                    default="ULTRA",
                    tooltip="VSR 模型档位；DENOISE/DEBLUR 档在原分辨率处理。",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="效果强度。"),
                io.Combo.Input(
                    "image_encoding",
                    options=list(_IMAGE_ENCODINGS),
                    default=_ENCODING_RGB8,
                    tooltip="SDK 像素编码：10-bit 保留更高精度（不负责 SDR→HDR 转换）。",
                ),
                io.Combo.Input(
                    "format",
                    options=["nvenc_h264-mp4", "nvenc_hevc-mp4", "h264-mp4"],
                    default="nvenc_h264-mp4",
                    tooltip="nvenc_* 为 NVIDIA 硬件编码（最快，推荐）；h264-mp4 为 libx264 软编（用 crf/preset）。",
                ),
                io.Combo.Input(
                    "pix_fmt",
                    options=["yuv420p", "yuv420p10le", "p010le"],
                    default="yuv420p",
                    tooltip="像素格式。10-bit：软编选 yuv420p10le，nvenc 选 p010le。",
                ),
                io.Int.Input("bitrate", default=15, min=1, max=999, step=1, tooltip="视频码率数值，仅 nvenc 格式使用。"),
                io.Boolean.Input("megabit", default=True, tooltip="True 时码率单位为 Mbps，False 为 Kbps；仅 nvenc 格式使用。"),
                io.Int.Input("crf", default=19, min=0, max=51, step=1, tooltip="恒定质量因子，仅 h264-mp4 软编使用。"),
                io.Combo.Input("preset", options=_PRESET_CHOICES, default="medium", tooltip="x264 速度/压缩比预设，仅 h264-mp4 软编使用。"),
            ],
            outputs=[
                io.Video.Output(display_name="video", tooltip="内存中的 MP4 视频（fragmented 容器，PyAV/官方节点完全兼容）。"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, fps: float, frames_per_chunk: int,
                scale: float, quality: str, strength: float, image_encoding: str,
                format: str, pix_fmt: str, bitrate: int, megabit: bool,
                crf: int, preset: str, audio: dict = None) -> io.NodeOutput:
        if nvvfx is None:
            raise RuntimeError(
                f"nvvfx 未安装（{_VSR_IMPORT_ERROR}）：RTX Video Super Resolution 需要 nvvfx 与 NVIDIA RTX GPU"
            )
        if images.dim() != 4 or images.shape[0] == 0:
            raise ValueError(f"images 应为非空 [N,H,W,C] 张量，实际 shape={tuple(images.shape)}")
        n_frames, height, width, channels = images.shape
        if channels not in (3, 4):
            raise ValueError(f"帧通道数应为 3(RGB) 或 4(RGBA)，实际为 {channels}")
        if channels == 4:
            images = images[..., :3]  # H.264 不承载 alpha，与 PSLVideoCombineMemory 行为一致
        if fps <= 0:
            raise ValueError(f"fps 必须为正数，实际为 {fps}")
        if frames_per_chunk < 1:
            raise ValueError(f"frames_per_chunk 必须 >= 1，实际为 {frames_per_chunk}")
        if not 1.0 <= scale <= 4.0:
            raise ValueError(f"scale 应在 [1.0, 4.0] 内，实际为 {scale}")
        if quality not in _VSR_QUALITY_OPTIONS:
            raise ValueError(f"未知 quality：{quality}，可选：{_VSR_QUALITY_OPTIONS}")
        if image_encoding not in _IMAGE_ENCODINGS:
            raise ValueError(f"未知 image_encoding：{image_encoding}，可选：{list(_IMAGE_ENCODINGS)}")

        if quality.startswith(("DENOISE_", "DEBLUR_")):
            out_w, out_h = width, height
        else:
            out_w = max(8, round(int(width * scale) / 8) * 8)
            out_h = max(8, round(int(height * scale) / 8) * 8)
        if format in _NVENC_FORMATS and (out_w % 2 or out_h % 2):
            raise ValueError(f"NVENC 要求偶数分辨率，当前输出 {out_w}x{out_h}")

        args = []
        proc = None
        readers = ()
        out_buf = err_buf = None
        tmp_audio = None
        started = time.perf_counter()
        try:
            args = _build_input_args(out_w, out_h, fps, 3)
            if audio is not None:
                tmp_audio, sample_rate, n_channels = _write_audio_raw(
                    audio, folder_paths.get_temp_directory()
                )
                args += [
                    "-f", "f32le", "-ar", str(sample_rate), "-ac", str(n_channels), "-i", tmp_audio,
                    "-map", "0:v", "-map", "1:a",
                ]
            args += _build_output_args(format, pix_fmt, bitrate, megabit, crf, preset)
            if audio is not None:
                args += [
                    "-c:a", "aac",
                    "-af", f"apad=whole_dur={n_frames / fps + 1}",
                    "-shortest",
                ]
            args += ["-movflags", "+frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]

            proc, readers, out_buf, err_buf = _start_encoder(args)
            pbar = comfy.utils.ProgressBar(n_frames)
            device, device_index = _effect_device()
            hop_frames = max(1, _MAX_BATCH_PIXELS // (out_w * out_h))
            n_chunks = (n_frames + frames_per_chunk - 1) // frames_per_chunk
            logger.info(
                "RTX 分片放大开始：%d 帧 %dx%d → %dx%d，每片 %d 帧（共 %d 片），quality=%s，scale=%.2f",
                n_frames, width, height, out_w, out_h, frames_per_chunk, n_chunks, quality, scale,
            )

            with torch.cuda.device(device), nvvfx.VideoSuperRes(
                quality=nvvfx.QualityLevel[quality],
                strength=strength,
                device=device_index,
                image_encoding=nvvfx.ImageEncoding.RGB10A2 if image_encoding == _ENCODING_RGB10A2 else nvvfx.ImageEncoding.RGB8,
            ) as effect:
                effect.output_width = out_w
                effect.output_height = out_h
                effect.load()

                for chunk_index, chunk_start in enumerate(
                    range(0, n_frames, frames_per_chunk), start=1
                ):
                    chunk_end = min(chunk_start + frames_per_chunk, n_frames)
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    for hop_start in range(chunk_start, chunk_end, hop_frames):
                        batch = images[hop_start:min(hop_start + hop_frames, chunk_end)].to(
                            device=device, dtype=torch.float32
                        )
                        if image_encoding == _ENCODING_RGB10A2:
                            prepared = _pack_rgb10a2(batch)  # [b,H,W] uint32
                        else:
                            prepared = batch.movedim(-1, 1).contiguous()  # [b,3,H,W]
                        for frame in prepared:
                            proc.stdin.write(_effect_frame_bytes(effect, frame, image_encoding))
                            pbar.update(1)
                        del frame, prepared, batch
                    # 该分片已全部编码进同一条 ffmpeg 管道：释放放大中转与显存
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.debug(
                        "分片 %d/%d 完成：帧 %d-%d", chunk_index, n_chunks, chunk_start, chunk_end - 1
                    )

            proc.stdin.flush()
            proc.stdin.close()
            out_buf = _finish_encoder(proc, readers, out_buf, err_buf, args)
            logger.info(
                "RTX 分片放大完成：%d 帧 → %dx%d，用时 %.1fs，视频 %.1f MB",
                n_frames, out_w, out_h, time.perf_counter() - started,
                out_buf.getbuffer().nbytes / 1e6,
            )
            return io.NodeOutput(InputImpl.VideoFromFile(out_buf))
        except BrokenPipeError as exc:
            if proc is not None:
                _terminate(proc, *readers)
            stderr_tail = err_buf.getvalue().decode(*ENCODE_ARGS) if err_buf is not None else ""
            raise RuntimeError(
                f"ffmpeg 提前退出（BrokenPipe），参数：{' '.join(args)}\n{stderr_tail}"
            ) from exc
        finally:
            if proc is not None:
                try:
                    if proc.stdin and not proc.stdin.closed:
                        proc.stdin.close()
                except Exception:
                    pass
                if proc.poll() is None:
                    _terminate(proc, *readers)
                for reader in readers:
                    if reader.is_alive():
                        reader.join(timeout=5)
            if tmp_audio is not None:
                try:
                    os.remove(tmp_audio)
                except OSError as exc:
                    logger.warning("临时音频文件删除失败：%s（%s）", tmp_audio, exc)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()