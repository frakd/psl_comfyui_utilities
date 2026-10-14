"""PSL 内存视频合成节点。

PSLVideoCombineMemory（Video Combine (Memory)）：
    帧 + 音频 → 内存 MP4，全程不落盘。编码引擎照搬 VHS VideoCombine 的
    本机优化版（当时会话《ComfyUI视频保存节点加速方案》定稿），三项加速机制
    全部保留：
      1. GPU 往返量化 —— CPU 大帧搬上 GPU 做乘/clamp/量化，只回传 uint8
         （实测 1403 帧 1280x2304 的转换从 ~26s 压到 ~8.5s，显存峰值恒定）；
      2. 有界队列流水线 —— 转换在 daemon 线程，经 maxsize=4 的队列交给
         主线程写 ffmpeg stdin，转换与 NVENC 编码重叠，队列满即背压；
      3. NVENC 硬编 + fake_trc 色彩参数 —— 默认 h264_nvenc / yuv420p /
         15 Mbps，输入按 sRGB 谎报为 BT.709（fake_trc），输出侧
         scale=out_color_matrix=bt709 消除 gamma 漂移。
    与 VHS 的唯一结构差异是输出与音频：
      - 输出：ffmpeg 以 fragmented MP4（-movflags +frag_keyframe+empty_moov）
        写 stdout（普通 MP4 muxer 需要 seek，管道写不了），Python 侧用
        BytesIO 接住后包成官方 io.Video（VideoFromFile(BytesIO)），可无缝
        接 ComfyUI 一切 VIDEO 下游（多段合并直接用官方 ConcatenateVideo，
        落盘用官方 SaveVideo，签名一致时均为流拷贝）；
      - 音频：VHS 是"先落盘无声视频 → 第二个 ffmpeg 进程 mux"，但 stdin 只
        有一条已被视频帧占用，因此这里把音频写成临时 f32le 裸流文件，与视频
        在同一条命令里完成 nvenc + aac + mux；apad + -shortest 对齐语义照搬。

已知限制：中间视频驻留内存（15 Mbps 约 112 MB/分钟/段），ComfyUI 输出缓存
会持有引用；长会话多段拼接时注意内存余量。
"""

import logging
import os
import queue
import shutil
import subprocess
import threading
import tempfile
from io import BytesIO

import torch
import folder_paths
from comfy_api.latest import io, InputImpl

import comfy.utils

logger = logging.getLogger("psl.video_memory")

ENCODE_ARGS = ("utf-8", "replace")

# 支持 Nvenc 的格式走 -b:v 码率控制；软编 h264-mp4 走 -crf/-preset。
_NVENC_FORMATS = {
    "nvenc_h264-mp4": ["-c:v", "h264_nvenc"],
    "nvenc_hevc-mp4": ["-c:v", "hevc_nvenc", "-vtag", "hvc1"],
}
_PRESET_CHOICES = ["medium", "ultrafast", "superfast", "veryfast", "faster", "fast", "slow", "slower", "veryslow"]

# 三个格式的 fake_trc 均为 bt709（与 VHS JSON 一致）：
# 输入是未定义色域的 RGB（实践中即 sRGB），对 ffmpeg 谎报为 BT.709 传递函数，
# 避免 sRGB→BT.709 的 gamma 漂移（YouTube 等平台按 full BT.709 解读）。
_FAKE_TRC = "bt709"

_FFMPEG_PATH = None


def _ffmpeg() -> str:
    """定位 ffmpeg（本机 full build 在 PATH 上），结果缓存。"""
    global _FFMPEG_PATH
    if _FFMPEG_PATH is None:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError(
                "未找到 ffmpeg：请确认已安装并在 PATH 中（本节点依赖外部 ffmpeg 子进程编码）"
            )
        _FFMPEG_PATH = found
    return _FFMPEG_PATH


def _frame_bytes(tensor: torch.Tensor) -> bytes:
    """单帧张量 → 原始 RGB(A) 字节。GPU 往返量化（VHS tensor_to_int 优化逻辑）。

    CPU 大帧（>=512x512x3）搬上 GPU 做乘/clamp/量化，只把小体积 uint8 传回；
    帧本身已在 GPU 或小张量时留在原地。clamp 显式截断（torch 整型转换是取模
    而非饱和，越界像素必须 clamp）。乘法已产生新张量，clamp_ 原地不会改到
    调用方的数据。
    """
    if (not tensor.is_cuda) and torch.cuda.is_available() and tensor.numel() >= 512 * 512 * 3:
        tensor = tensor.cuda()
    q = (tensor * 255.0 + 0.5).clamp_(0, 255).to(torch.uint8)
    if tensor.is_cuda:
        del tensor  # 先释放 32-bit GPU 副本，再回传小体积 uint8
    return q.cpu().numpy().tobytes()


def _drain_queue(q: queue.Queue) -> None:
    """清空队列，解阻塞可能卡在 put() 上的生产者线程（VHS 同款）。"""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _terminate(proc: subprocess.Popen, *threads: threading.Thread) -> None:
    try:
        proc.kill()
    except Exception:
        pass
    for t in threads:
        if t is not None:
            t.join(timeout=5)


def _encode_to_memory(args: list, images, total_frames: int, pbar) -> BytesIO:
    """流式喂帧给 ffmpeg，从 stdout 收 fragmented MP4 到内存。

    线程结构：
      - 生产者（daemon 线程）：张量→字节转换（GPU 量化），经有界队列交给主线程；
        异常对象直接放进队列带回主线程——保存失败必须显式报错，不能像 VHS 那样
        在线程里静默失败后产出残缺文件。
      - 主线程：从队列取字节写 ffmpeg stdin（BrokenPipe = ffmpeg 提前退出）。
      - stdout/stderr 各一个 drain 线程：不并发读取的话管道缓冲写满 ffmpeg
        会卡死；stderr 也 drain 掉，避免 -v error 之外的输出积压。
    """
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_buf = BytesIO()
    err_buf = BytesIO()

    def _drain(pipe, sink):
        try:
            while True:
                chunk = pipe.read(1 << 20)
                if not chunk:
                    break
                sink.write(chunk)
        except Exception:
            pass  # 失败判定以 returncode + stderr 为准，这里不吞主错误

    reader_out = threading.Thread(target=_drain, args=(proc.stdout, out_buf), name="psl-ffmpeg-stdout", daemon=True)
    reader_err = threading.Thread(target=_drain, args=(proc.stderr, err_buf), name="psl-ffmpeg-stderr", daemon=True)
    reader_out.start()
    reader_err.start()

    sentinel_done = object()
    q = queue.Queue(maxsize=4)

    def produce():
        try:
            for image in images:
                pbar.update(1)
                q.put(_frame_bytes(image))
        except BaseException as exc:
            q.put(exc)
            return
        q.put(sentinel_done)

    producer = threading.Thread(target=produce, name="psl-frame-feed", daemon=True)
    producer.start()

    producer_error = None
    pipe_broken = False
    try:
        while True:
            item = q.get()
            if item is sentinel_done:
                break
            if isinstance(item, BaseException):
                producer_error = item
                break
            proc.stdin.write(item)
        if producer_error is None:
            proc.stdin.flush()
            proc.stdin.close()
    except BrokenPipeError:
        pipe_broken = True
    except BaseException:
        # 主线程自身异常（如执行中断）：解阻塞生产者、终止 ffmpeg、原样抛出
        _drain_queue(q)
        _terminate(proc, reader_out, reader_err, producer)
        raise

    if producer_error is not None or pipe_broken:
        _drain_queue(q)
        try:
            proc.stdin.close()
        except Exception:
            pass
    producer.join(timeout=5)
    if producer_error is not None:
        _terminate(proc, reader_out, reader_err)
        raise producer_error

    reader_out.join(timeout=30)  # stdout 到 EOF 即 ffmpeg 已完成输出
    proc.wait(timeout=30)
    stderr_tail = err_buf.getvalue().decode(*ENCODE_ARGS)
    if proc.returncode != 0 or out_buf.tell() == 0:
        raise RuntimeError(
            f"ffmpeg 编码失败 (exit={proc.returncode})，参数：{' '.join(args)}\n{stderr_tail}"
        )
    if stderr_tail:
        logger.warning("ffmpeg 输出了一些诊断信息：\n%s", stderr_tail)
    out_buf.seek(0)
    return out_buf


def _build_input_args(width: int, height: int, fps: float, channels: int) -> list:
    """ffmpeg 输入侧参数（rawvideo 裸帧，sRGB 谎报 BT.709，逐项对齐 VHS）。"""
    return [
        _ffmpeg(), "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba" if channels == 4 else "rgb24",
        "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
        "-color_trc", _FAKE_TRC,
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
    ]


def _build_output_args(fmt: str, pix_fmt: str, bitrate: int, megabit: bool,
                       crf: int, preset: str) -> list:
    """视频编码输出参数。必须放在所有 -i 之后：ffmpeg 把输出选项之前的
    选项归给最近的输入文件，混排会把 -b:v 之类误绑到音频输入上。"""
    if fmt in _NVENC_FORMATS:
        args = list(_NVENC_FORMATS[fmt])
        args += ["-b:v", f"{int(bitrate)}M" if megabit else f"{int(bitrate)}K"]
    elif fmt == "h264-mp4":
        args = ["-c:v", "libx264", "-crf", str(int(crf)), "-preset", preset]
    else:
        raise ValueError(f"不支持的编码格式：{fmt}")
    args += [
        "-pix_fmt", pix_fmt,
        "-vf", "scale=out_color_matrix=bt709",
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    ]
    return args


def _write_audio_raw(audio: dict, directory: str) -> tuple:
    """AUDIO 波形 → 临时 f32le 交错裸流文件（与 VHS 的 stdin 喂法同一格式）。

    返回 (临时文件路径, 声道数)。文件由调用方在 finally 中删除。
    """
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() != 3 or waveform.shape[0] != 1:
        raise ValueError(f"音频波形形状应为 [1, C, N]，实际为 {tuple(waveform.shape)}")
    data = waveform[0].transpose(0, 1).contiguous().cpu().numpy()  # [N, C] 交错
    channels = int(data.shape[1])
    fd, path = tempfile.mkstemp(suffix=".f32le", prefix="psl_audio_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.astype("<f4", copy=False).tobytes())
    except BaseException:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path, sample_rate, channels


class PSLVideoCombineMemory(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PSLVideoCombineMemory",
            display_name="Video Combine (Memory)",
            category="video/psl",
            description=(
                "帧 + 音频直接在内存中合成视频（不落盘）。编码参数为本机优化方案："
                "NVENC 硬编 + GPU 量化 + 流水线喂帧。输出官方 VIDEO 类型，可直接接"
                "合并落盘节点或官方 SaveVideo。"
            ),
            inputs=[
                io.Image.Input("images", tooltip="视频帧 [N,H,W,C]，float32 [0,1]。4 通道时 alpha 会被丢弃（H.264 不支持透明）。"),
                io.Audio.Input("audio", optional=True, tooltip="可选音轨，编码为 AAC，与视频时长对齐（apad + shortest）。"),
                io.Float.Input("fps", default=30.0, min=1.0, max=240.0, step=0.01, tooltip="输出帧率。"),
                io.Combo.Input(
                    "format",
                    options=["nvenc_h264-mp4", "nvenc_hevc-mp4", "h264-mp4"],
                    default="nvenc_h264-mp4",
                    tooltip="nvenc_* 为 NVIDIA 硬件编码（本机最快，推荐）；h264-mp4 为 libx264 软编（用 crf/preset）。"
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
            outputs=[io.Video.Output(display_name="video", tooltip="内存中的 MP4 视频（fragmented 容器，PyAV/官方节点完全兼容）。")],
        )

    @classmethod
    def execute(cls, images, fps, format, bitrate, megabit,
                audio=None, pix_fmt="yuv420p", crf=19, preset="medium") -> io.NodeOutput:
        if images.dim() != 4 or images.shape[0] == 0:
            raise ValueError(f"images 应为非空 [N,H,W,C] 张量，实际 shape={tuple(images.shape)}")
        n_frames, height, width, channels = images.shape
        if channels not in (3, 4):
            raise ValueError(f"帧通道数应为 3(RGB) 或 4(RGBA)，实际为 {channels}")
        if (width % 2 or height % 2) and format in _NVENC_FORMATS:
            raise ValueError(f"NVENC 要求偶数分辨率，当前 {width}x{height}，请在上游调整尺寸")
        if fps <= 0:
            raise ValueError(f"fps 必须为正数，实际为 {fps}")

        tmp_audio = None
        try:
            args = _build_input_args(width, height, fps, channels)
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

            pbar = comfy.utils.ProgressBar(n_frames)
            buffer = _encode_to_memory(args, images, n_frames, pbar)
            return io.NodeOutput(InputImpl.VideoFromFile(buffer))
        finally:
            if tmp_audio is not None:
                try:
                    os.remove(tmp_audio)
                except OSError as exc:
                    logger.warning("临时音频文件删除失败：%s（%s）", tmp_audio, exc)
