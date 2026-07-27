import math
import torch
import torchaudio
import warnings
import logging
from fractions import Fraction
from comfy_api.latest import io, InputImpl, Types
import comfy.utils
import torch


class ImageSizeTransformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "target_height": ("INT", {"default": 720, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "aspect_ratio",)
    FUNCTION = "transform"
    CATEGORY = "SlowAI"

    def transform(self, image, target_width, target_height):
        # 1. 获取原图尺寸
        _, h, w, _ = image.shape
        
        # 2. 计算最接近的比例
        ratios = {
            "1:1": 1.0, "2:3": 2/3, "3:2": 3/2, "3:4": 3/4, 
            "4:3": 4/3, "4:5": 4/5, "5:4": 5/4, "9:16": 9/16, 
            "16:9": 16/9, "21:9": 21/9
        }
        target_ratio = target_width / target_height
        closest_ratio = min(ratios.keys(), key=lambda k: abs(ratios[k] - target_ratio))

        # 3. 生成中文提示词逻辑
        is_orientation_change = (w > h and target_width < target_height) or (w < h and target_width > target_height)
        
        if is_orientation_change:
            instruction = (f"将原图从 {w}x{h} 调整为 {target_width}x{target_height}。"
                           f"这是一次横竖版式转换，请在改变尺寸的同时，严格保持原图的视觉风格、主体内容和构图逻辑一致，避免画面拉伸或内容丢失。\n")
        else:
            instruction = (f"将原图从 {w}x{h} 调整为 {target_width}x{target_height}。"
                           f"这是一次缩放处理，请在改变尺寸的同时，确保画面清晰，并完美保留原图的风格、细节和主体内容。\n")

        return (instruction, closest_ratio)


class TextSplitter:
    """
    将输入文本按指定分隔符切分为最多5段，不足的部分返回空字符串。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "separator": ("STRING", {
                    "multiline": False,
                    "default": ","
                }),
            },
        }

    RETURN_TYPES = ("STRING",) * 5
    RETURN_NAMES = tuple(f"segment_{i+1}" for i in range(5))
    FUNCTION = "split"
    CATEGORY = "SlowAI"

    def split(self, text, separator):
        # 按分隔符切分，maxsplit=4 得到最多5个元素
        parts = text.split(separator, 4)

        # 补足到5个，不足部分为空字符串
        if len(parts) < 5:
            parts.extend([""] * (5 - len(parts)))

        return tuple(parts[:5])


def match_audio_sample_rates(waveform_1, sr1, waveform_2, sr2):
    """重采样至较高采样率，使两个音频波形可拼接"""
    if sr1 == sr2:
        return waveform_1, waveform_2, sr1
    if sr1 > sr2:
        waveform_2 = torchaudio.functional.resample(waveform_2, sr2, sr1)
        logging.info(f"VideoConcat: resample audio2 from {sr2}Hz to {sr1}Hz")
        return waveform_1, waveform_2, sr1
    else:
        waveform_1 = torchaudio.functional.resample(waveform_1, sr1, sr2)
        logging.info(f"VideoConcat: resample audio1 from {sr1}Hz to {sr2}Hz")
        return waveform_1, waveform_2, sr2


class VideoConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoConcat",
            display_name="Concatenate Videos",
            category="image/video",
            description="将两个视频的帧和音频按顺序拼接",
            inputs=[
                io.Video.Input("video_a", tooltip="第一个视频"),
                io.Video.Input("video_b", tooltip="第二个视频"),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    def execute(cls, video_a, video_b) -> io.NodeOutput:
        # ---------- 提取组件 ----------
        comp_a = video_a.get_components()  # Types.VideoComponents
        comp_b = video_b.get_components()

        images_a = comp_a.images          # [T, H, W, C]
        fps_a = float(comp_a.frame_rate) if comp_a.frame_rate else 30.0
        audio_a = comp_a.audio            # dict: {"waveform": [B, C, samples], "sample_rate": int} or None

        images_b = comp_b.images
        fps_b = float(comp_b.frame_rate) if comp_b.frame_rate else 30.0
        audio_b = comp_b.audio

        # ---------- 分辨率检查 ----------
        if images_a.shape[1:] != images_b.shape[1:]:
            raise ValueError(
                f"视频分辨率不一致: {images_a.shape[1:]} vs {images_b.shape[1:]}"
            )

        # ---------- 帧率警告 ----------
        if abs(fps_a - fps_b) > 1e-4:
            warnings.warn(
                f"输入视频帧率不同: {fps_a} vs {fps_b}，输出将采用第一个视频的帧率 ({fps_a})。"
            )

        # ---------- 拼接图像 ----------
        concatenated_images = torch.cat([images_a, images_b], dim=0)

        # ---------- 拼接音频 ----------
        concatenated_audio = None
        if audio_a is not None and audio_b is not None:
            wf_a = audio_a["waveform"]   # [B, C, T]
            sr_a = audio_a["sample_rate"]
            wf_b = audio_b["waveform"]
            sr_b = audio_b["sample_rate"]

            # 处理通道数不匹配：单声道复制为立体声，若双方通道数不同且都不是单声道则放弃拼接
            channels_a = wf_a.shape[1]
            channels_b = wf_b.shape[1]

            if channels_a != channels_b:
                if channels_a == 1:
                    wf_a = wf_a.repeat(1, channels_b, 1)
                    logging.info("VideoConcat: audio_a 单声道扩展为 %d 声道", channels_b)
                elif channels_b == 1:
                    wf_b = wf_b.repeat(1, channels_a, 1)
                    logging.info("VideoConcat: audio_b 单声道扩展为 %d 声道", channels_a)
                else:
                    warnings.warn(
                        f"音频通道数不匹配 ({channels_a} vs {channels_b})，无法自动处理，将仅保留第一个视频的音频。"
                    )
                    concatenated_audio = audio_a
                    # 跳过拼接逻辑
                    audio_a = None
                    audio_b = None

            if audio_a is not None and audio_b is not None:
                # 采样率统一
                wf_a, wf_b, out_sr = match_audio_sample_rates(wf_a, sr_a, wf_b, sr_b)
                # 在时间维度拼接 (dim=-1)
                concatenated_wf = torch.cat([wf_a, wf_b], dim=-1)
                concatenated_audio = {"waveform": concatenated_wf, "sample_rate": out_sr}
        elif audio_a is not None:
            concatenated_audio = audio_a
        elif audio_b is not None:
            concatenated_audio = audio_b

        # ---------- 构造输出视频 ----------
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=concatenated_images,
                audio=concatenated_audio,
                frame_rate=Fraction(fps_a)
            )
        )
        return io.NodeOutput(output_video)


class VideoSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoSplit",
            display_name="Video Split",
            category="image/video",
            description=(
                "把视频按固定时长切成最多 12 段。最后一个端口（第12段）会收进所有剩余内容。"
                "未用到的端口输出空视频。可选是否保留音轨。"
            ),
            inputs=[
                io.Video.Input("video", tooltip="待切分的视频"),
                io.Float.Input(
                    "seconds_per_segment",
                    default=15.0,
                    min=0.1,
                    max=600.0,
                    step=0.1,
                    tooltip="每段时长（秒）",
                ),
                io.Boolean.Input(
                    "keep_audio",
                    default=False,
                    tooltip=(
                        "是否保留音轨。送往 Seedance 等视频生成 API 时建议关闭，"
                        "避免触发音频版权检测。"
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(display_name="segment_1"),
                io.Video.Output(display_name="segment_2"),
                io.Video.Output(display_name="segment_3"),
                io.Video.Output(display_name="segment_4"),
                io.Video.Output(display_name="segment_5"),
                io.Video.Output(display_name="segment_6"),
                io.Video.Output(display_name="segment_7"),
                io.Video.Output(display_name="segment_8"),
                io.Video.Output(display_name="segment_9"),
                io.Video.Output(display_name="segment_10"),
                io.Video.Output(display_name="segment_11"),
                io.Video.Output(display_name="segment_12"),
            ],
        )

    @staticmethod
    def _make_segment(images, start_f, end_f, fps, audio, sample_rate, keep_audio):
        seg_imgs = images[start_f:end_f]

        seg_audio = None
        if keep_audio and audio is not None and sample_rate:
            a_start = int(start_f / fps * sample_rate)
            a_end = int(end_f / fps * sample_rate)
            seg_audio = {
                "waveform": audio["waveform"][:, :, a_start:a_end],
                "sample_rate": sample_rate,
            }

        return InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=seg_imgs,
                audio=seg_audio,
                frame_rate=fps,
            )
        )

    @classmethod
    def execute(cls, video, seconds_per_segment, keep_audio) -> io.NodeOutput:
        MAX_SEG = 12

        comp = video.get_components()
        images = comp.images                 # [T,H,W,C]
        fps = comp.frame_rate
        audio = comp.audio
        sample_rate = audio["sample_rate"] if audio is not None else None

        total_frames = images.shape[0]
        frames_per_seg = max(1, int(round(float(fps) * seconds_per_segment)))

        empty_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images[0:0],
                audio=None,
                frame_rate=fps,
            )
        )

        segments = []
        start_f = 0
        while start_f < total_frames and len(segments) < MAX_SEG:
            is_last_slot = (len(segments) == MAX_SEG - 1)
            end_f = total_frames if is_last_slot else min(start_f + frames_per_seg, total_frames)

            segments.append(
                cls._make_segment(images, start_f, end_f, fps, audio, sample_rate, keep_audio)
            )
            start_f = end_f

        while len(segments) < MAX_SEG:
            segments.append(empty_video)

        return io.NodeOutput(*segments)

class VideoDownsize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoDownsize",
            display_name="Video Downsize",
            category="image/video",
            description=(
                "按最大总像素数等比缩小视频，保持宽高比、边长对齐偶数。"
                "未超限则原样返回。用于满足 Seedance 等模型的分辨率上限。"
            ),
            inputs=[
                io.Video.Input("video", tooltip="待缩放的视频"),
                io.Int.Input(
                    "max_pixels",
                    default=927408,
                    min=4096,
                    max=927408,
                    step=16,
                    tooltip="允许的最大总像素数（宽×高）。默认 927408 对应 Seedance。",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["lanczos", "bicubic", "bilinear", "area", "nearest-exact"],
                    default="lanczos",
                    tooltip="缩放算法",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    def execute(cls, video, max_pixels, upscale_method) -> io.NodeOutput:
        comp = video.get_components()
        images = comp.images                 # [T,H,W,C]
        fps = comp.frame_rate
        audio = comp.audio

        # 空视频 / 未超限：原样返回
        if images.shape[0] == 0:
            return io.NodeOutput(video)

        T, H, W, C = images.shape
        cur = H * W
        if cur <= max_pixels:
            return io.NodeOutput(video)

        scale = math.sqrt(max_pixels / cur)
        new_w = int(math.floor(W * scale))
        new_h = int(math.floor(H * scale))
        new_w -= new_w % 2
        new_h -= new_h % 2
        new_w = max(2, new_w)
        new_h = max(2, new_h)

        # 兜底：floor+偶数化后仍可能极端超限
        while new_w * new_h > max_pixels:
            if new_w >= new_h:
                new_w -= 2
            else:
                new_h -= 2

        imgs = images.movedim(-1, 1)  # [T,C,H,W]
        imgs = comfy.utils.common_upscale(imgs, new_w, new_h, upscale_method, "disabled")
        imgs = imgs.movedim(1, -1)    # [T,H,W,C]

        logging.info(
            "VideoDownsize: %dx%d(%d px) -> %dx%d(%d px)",
            W, H, cur, new_w, new_h, new_w * new_h,
        )

        out_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=imgs,
                audio=audio,
                frame_rate=fps,
            )
        )
        return io.NodeOutput(out_video)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageSizeTransformer": ImageSizeTransformer,
    "TextSplitter": TextSplitter,
    "VideoSplit": VideoSplit,
    "VideoConcat": VideoConcat,
    "VideoDownsize": VideoDownsize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSizeTransformer": "ImageSizeTransformer",
    "TextSplitter": "TextSplitter",
    "VideoSplit": "VideoSplit",
    "VideoConcat": "VideoConcat",
    "VideoDownsize": "VideoDownsize",
}