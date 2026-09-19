# coding=utf-8
"""ComfyUI Wan I2V / S2V video generation."""
from __future__ import annotations

import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from config import (
    COMFY_URL,
    FPS,
    I2V_CLIP,
    I2V_UNET,
    I2V_VAE,
    INPUT_DIR,
    OUTPUT_DIR,
    S2V_AUDIO_ENC,
    S2V_CLIP,
    S2V_UNET,
    S2V_VAE,
    UPLOAD_DIR,
)

LogCb = Callable[[str, str, dict[str, Any] | None], None]

I2V_UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
I2V_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
I2V_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"

WAN_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，"
    "静止不动的画面，畸形的，多余的手指，静止, static, frozen, still image, no motion, "
    "ghosting, double chin, extra jaw, overlapping faces, motion smear, duplicated mouth"
)


def _log(cb: LogCb | None, level: str, msg: str, data: dict[str, Any] | None = None) -> None:
    if callable(cb):
        cb(level, msg, data)


def _copy_to_input(src: Path) -> str:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = INPUT_DIR / src.name
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return src.name


def _gen_size(width: int, height: int) -> tuple[int, int]:
    """Wan 14B 用 720x1280 或 480x832，避免 1080x1920 爆显存。"""
    if height >= width:
        if min(width, height) >= 720:
            return 720, 1280
        return 480, 832
    if min(width, height) >= 720:
        return 1280, 720
    return 832, 480


def _frame_length(duration_s: float, fps: int = FPS, max_len: int = 49) -> int:
    n = int(round(max(duration_s, 1.0) * fps))
    n = max(17, n)
    n = ((n - 1) // 4) * 4 + 1
    return min(n, max_len)


def _wait_video(prompt_id: str, timeout: int = 600, log_callback: LogCb | None = None) -> Path:
    deadline = time.time() + timeout
    poll = 0
    while time.time() < deadline:
        poll += 1
        r = httpx.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        entry = r.json().get(prompt_id, {})
        outputs = entry.get("outputs", {})
        for node_out in outputs.values():
            for key in ("videos", "gifs", "images"):
                for item in node_out.get(key) or []:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get("filename")
                    if not filename:
                        continue
                    sub = item.get("subfolder") or ""
                    path = OUTPUT_DIR / sub / filename if sub else OUTPUT_DIR / filename
                    if path.exists():
                        _log(log_callback, "info", "ComfyUI 视频输出就绪", {"path": str(path), "poll": poll})
                        return path
        status = entry.get("status") or {}
        if status.get("status_str") == "error" or status.get("status_msg") == "error":
            raise RuntimeError(f"ComfyUI video failed: {status}")
        time.sleep(2)
    raise TimeoutError(f"ComfyUI video timed out after {timeout}s prompt_id={prompt_id}")


def _mux_scale(src: Path, dst: Path, width: int, height: int, audio_path: Path | None, duration_s: float) -> None:
    # 不要 fps=30：16→30 会插帧/复帧，下巴轮廓容易出现重影。
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if audio_path and audio_path.exists():
        cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _build_i2v_workflow(
    image_filename: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    length: int,
    seed: int,
) -> dict[str, Any]:
    neg = negative or WAN_NEG
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": I2V_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": I2V_LORA_HIGH, "strength_model": 1.0}},
        "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 8.0}},
        "4": {"class_type": "UNETLoader", "inputs": {"unet_name": I2V_UNET_LOW, "weight_dtype": "default"}},
        "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": I2V_LORA_LOW, "strength_model": 1.0}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 8.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": I2V_CLIP, "type": "wan", "device": "default"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": neg}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": I2V_VAE}},
        "11": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "12": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "13": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["12", 0], "image": ["11", 0], "crop": "center"}},
        "14": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["8", 0],
                "negative": ["9", 0],
                "vae": ["10", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "clip_vision_output": ["13", 0],
                "start_image": ["11", 0],
            },
        },
        "15": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["3", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["14", 0],
                "negative": ["14", 1],
                "latent_image": ["14", 2],
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
            },
        },
        "16": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["6", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["14", 0],
                "negative": ["14", 1],
                "latent_image": ["15", 0],
                "start_at_step": 2,
                "end_at_step": 4,
                "return_with_leftover_noise": "disable",
            },
        },
        "17": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["16", 0],
                "vae": ["10", 0],
                "tile_size": 768,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["17", 0], "fps": float(FPS)}},
        "19": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["18", 0], "filename_prefix": "console_i2v", "format": "mp4", "codec": "h264"},
        },
    }


def _build_s2v_workflow(
    image_filename: str,
    audio_filename: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    length: int,
    seed: int,
) -> dict[str, Any]:
    neg = negative or WAN_NEG
    # 下巴重影主因：4 步蒸馏 LoRA + CFG6/10 步过引导，以及 ffmpeg fps=30 插帧。
    # 3 秒片 < Wan 原生 81 帧，加 WanContextWindowsManual 重叠窗口会把两段口型金字塔融合，更容易双下巴。
    # 运动幅度用 ModelSamplingSD3 shift（8→5），时序用 VAEDecodeTiled temporal_overlap，而不是短片强制开窗。
    workflow: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": S2V_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 5.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": S2V_CLIP, "type": "wan", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": neg}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": S2V_VAE}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "8": {"class_type": "LoadAudio", "inputs": {"audio": audio_filename}},
        "9": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": S2V_AUDIO_ENC}},
        "10": {"class_type": "AudioEncoderEncode", "inputs": {"audio_encoder": ["9", 0], "audio": ["8", 0]}},
        "11": {
            "class_type": "WanSoundImageToVideo",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "vae": ["6", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "audio_encoder_output": ["10", 0],
                "ref_image": ["7", 0],
            },
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "seed": seed,
                "steps": 16,
                "cfg": 3.0,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["11", 2],
                "denoise": 1.0,
            },
        },
        "13": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["6", 0],
                "tile_size": 768,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        },
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["8", 0], "fps": float(FPS)}},
        "15": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["14", 0], "filename_prefix": "console_s2v", "format": "mp4", "codec": "h264"},
        },
    }
    if length > 81:
        workflow["16"] = {
            "class_type": "WanContextWindowsManual",
            "inputs": {
                "model": ["2", 0],
                "context_length": 81,
                "context_overlap": 16,
                "context_schedule": "standard_uniform",
                "context_stride": 1,
                "closed_loop": False,
                "fuse_method": "pyramid",
            },
        }
        workflow["12"]["inputs"]["model"] = ["16", 0]
    return workflow


def generate_shot_video(
    image_name: str,
    audio_name: str | None,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    duration_s: float,
    seed: int | None = None,
    mode: str = "i2v",
    log_callback: LogCb | None = None,
) -> Path:
    """Run Wan I2V or S2V in ComfyUI. Returns path of raw Comfy output video."""
    image_path = UPLOAD_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Shot image not found: {image_path}")
    image_filename = _copy_to_input(image_path)
    gen_w, gen_h = _gen_size(width, height)
    length = _frame_length(duration_s)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    prompt = positive_prompt or "a person moving naturally, cinematic motion"
    use_s2v = mode == "s2v" and bool(audio_name)
    if use_s2v:
        audio_path = UPLOAD_DIR / audio_name
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        audio_filename = _copy_to_input(audio_path)
        workflow = _build_s2v_workflow(image_filename, audio_filename, prompt, negative_prompt, gen_w, gen_h, length, seed)
        _log(
            log_callback,
            "info",
            "提交 ComfyUI S2V 工作流",
            {
                "size": f"{gen_w}x{gen_h}",
                "length": length,
                "steps": 16,
                "cfg": 3.0,
                "shift": 5.0,
                "lora": False,
                "context_windows": length > 81,
                "vae_temporal_overlap": 8,
            },
        )
    else:
        workflow = _build_i2v_workflow(image_filename, prompt, negative_prompt, gen_w, gen_h, length, seed)
        _log(log_callback, "info", "提交 ComfyUI I2V 工作流", {"size": f"{gen_w}x{gen_h}", "length": length, "unet": I2V_UNET})

    r = httpx.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=60)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"ComfyUI rejected prompt: {body}")
    prompt_id = body["prompt_id"]
    _log(log_callback, "info", "ComfyUI 视频任务已提交", {"prompt_id": prompt_id, "mode": "s2v" if use_s2v else "i2v"})
    return _wait_video(prompt_id, timeout=900, log_callback=log_callback)


def finalize_video(
    raw_path: Path,
    out_path: Path,
    width: int,
    height: int,
    audio_name: str | None,
    duration_s: float,
) -> None:
    audio_path = UPLOAD_DIR / audio_name if audio_name else None
    _mux_scale(raw_path, out_path, width, height, audio_path if audio_path and audio_path.exists() else None, duration_s)
