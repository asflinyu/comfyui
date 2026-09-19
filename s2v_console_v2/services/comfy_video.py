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

from services.comfy_runtime import GPU
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
    UPSCALE_MODEL,
    UPLOAD_DIR,
)

LogCb = Callable[[str, str, dict[str, Any] | None], None]

I2V_UNET_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
I2V_UNET_FP16 = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
I2V_UNET_LOW_FP16 = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
I2V_CLIP_FP16 = "umt5_xxl_fp16.safetensors"
I2V_LORA_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
I2V_LORA_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
T2V_UNET = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
T2V_UNET_LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
T2V_UNET_FP16 = "wan2.2_t2v_high_noise_14B_fp16.safetensors"
T2V_UNET_LOW_FP16 = "wan2.2_t2v_low_noise_14B_fp16.safetensors"
T2V_LORA_HIGH = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
T2V_LORA_LOW = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
CONSOLE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
HQ_I2V_STEPS = 20
HQ_I2V_SWITCH = 18  # Wan2.2 MoE boundary ≈ 0.875
HQ_S2V_STEPS = 24
HQ_CFG = 3.5

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


def _gpu_info() -> dict[str, Any]:
    """Read GPU 0 (Comfy 默认卡) name and VRAM. Multi-GPU 只看单卡，因为一条 Wan 不会自动拆卡。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return {"name": "unknown", "vram_gb": 32.0, "count": 0, "all": []}
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            vram_gb = float(parts[-1]) / 1024.0
        except ValueError:
            continue
        gpus.append({"name": parts[0], "vram_gb": round(vram_gb, 1)})
    if not gpus:
        return {"name": "unknown", "vram_gb": 32.0, "count": 0, "all": []}
    top = max(gpus, key=lambda g: g["vram_gb"])
    return {"name": top["name"], "vram_gb": top["vram_gb"], "count": len(gpus), "all": gpus}


def _select_t2v_profile(vram_gb: float) -> dict[str, Any]:
    """Wan 2.2 T2V 14B：纯文字，固定 fp8，不加载 fp16。"""
    if vram_gb >= 45:
        return {
            "tier": "high",
            "label": "T2V fp8 原生1080",
            "max_short": 1080,
            "unet_high": T2V_UNET,
            "unet_low": T2V_UNET_LOW,
            "clip": I2V_CLIP,
            "native_1080": True,
        }
    return {
        "tier": "lite",
        "label": "T2V fp8 720+超分",
        "max_short": 720,
        "unet_high": T2V_UNET,
        "unet_low": T2V_UNET_LOW,
        "clip": I2V_CLIP,
        "native_1080": False,
    }


def _select_profile(vram_gb: float) -> dict[str, Any]:
    """HQ 档按单卡显存升档。fp16 双专家约 54GB 权重，原生 1080 还要激活值。"""
    if vram_gb >= 70:
        return {
            "tier": "max",
            "label": "fp16 原生1080",
            "max_short": 1080,
            "i2v_unet": I2V_UNET_FP16,
            "i2v_unet_low": I2V_UNET_LOW_FP16,
            "clip": I2V_CLIP_FP16 if vram_gb >= 80 else I2V_CLIP,
            "native_1080": True,
        }
    if vram_gb >= 45:
        return {
            "tier": "high",
            "label": "fp8 原生1080",
            "max_short": 1080,
            "i2v_unet": I2V_UNET,
            "i2v_unet_low": I2V_UNET_LOW,
            "clip": I2V_CLIP,
            "native_1080": True,
        }
    return {
        "tier": "lite",
        "label": "fp8 720+超分",
        "max_short": 720,
        "i2v_unet": I2V_UNET,
        "i2v_unet_low": I2V_UNET_LOW,
        "clip": I2V_CLIP,
        "native_1080": False,
    }


def gpu_status() -> dict[str, Any]:
    info = _gpu_info()
    profile = _select_profile(float(info["vram_gb"]))
    t2v = _select_t2v_profile(float(info["vram_gb"]))
    return {
        "name": info["name"],
        "vram_gb": info["vram_gb"],
        "count": info["count"],
        "hq_tier": profile["tier"],
        "hq_label": profile["label"],
        "t2v_hq_label": t2v["label"],
        "t2v_native_1080": t2v["native_1080"],
    }


def _gen_size(width: int, height: int, max_short: int = 720) -> tuple[int, int]:
    """Wan 边长需 16 对齐。1080×1920 -> 1088×1920。小卡仍锁 720。"""
    portrait = height >= width
    req_short = min(width, height)
    if req_short < 720:
        return (480, 832) if portrait else (832, 480)
    if max_short >= 1080 and req_short >= 1080:
        return (1088, 1920) if portrait else (1920, 1088)
    return (720, 1280) if portrait else (1280, 720)


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


def _video_out_nodes(
    images_node: str,
    prefix: str,
    fps: float = float(FPS),
    audio_node: str | None = None,
    upscale: bool = False,
) -> dict[str, Any]:
    """Save mp4 from image batch; optional RealESRGAN x2 before encode."""
    nodes: dict[str, Any] = {}
    img_src: list[Any] = [images_node, 0]
    if upscale:
        nodes["u1"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}}
        nodes["u2"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["u1", 0], "image": img_src},
        }
        img_src = ["u2", 0]
    create_in: dict[str, Any] = {"images": img_src, "fps": fps}
    if audio_node:
        create_in["audio"] = [audio_node, 0]
    nodes["v1"] = {"class_type": "CreateVideo", "inputs": create_in}
    nodes["v2"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["v1", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"},
    }
    return nodes


def _mux_scale(src: Path, dst: Path, width: int, height: int, audio_path: Path | None, duration_s: float) -> None:
    # 锁 16fps，禁止默认 25/30 插帧，否则口型会被复帧看呆。
    vf = (
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p"
    )
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if audio_path and audio_path.exists():
        cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += [
        "-vf", vf, "-r", str(int(FPS)), "-vsync", "cfr",
        "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _build_i2v_workflow(
    image_filename: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    quality: str = "fast",
    upscale: bool = False,
    unet_high: str | None = None,
    unet_low: str | None = None,
    clip_name: str | None = None,
) -> dict[str, Any]:
    neg = negative or WAN_NEG
    hq = quality == "hq"
    unet_high = unet_high or I2V_UNET
    unet_low = unet_low or I2V_UNET_LOW
    clip_name = clip_name or I2V_CLIP
    if hq:
        high_model: list[Any] = ["2", 0]
        low_model: list[Any] = ["4", 0]
        steps = HQ_I2V_STEPS
        switch = HQ_I2V_SWITCH
        cfg = HQ_CFG
        workflow: dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_high, "weight_dtype": "default"}},
            "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
            "3": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_low, "weight_dtype": "default"}},
            "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 8.0}},
        }
    else:
        high_model = ["3", 0]
        low_model = ["6", 0]
        steps = 4
        switch = 2
        cfg = 1.0
        workflow = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": I2V_UNET, "weight_dtype": "default"}},
            "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": I2V_LORA_HIGH, "strength_model": 1.0}},
            "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 8.0}},
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": I2V_UNET_LOW, "weight_dtype": "default"}},
            "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": I2V_LORA_LOW, "strength_model": 1.0}},
            "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 8.0}},
        }
    workflow.update(
        {
            "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}},
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
                    "model": high_model,
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["14", 0],
                    "negative": ["14", 1],
                    "latent_image": ["14", 2],
                    "start_at_step": 0,
                    "end_at_step": switch,
                    "return_with_leftover_noise": "enable",
                },
            },
            "16": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": low_model,
                    "add_noise": "disable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["14", 0],
                    "negative": ["14", 1],
                    "latent_image": ["15", 0],
                    "start_at_step": switch,
                    "end_at_step": steps,
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
        }
    )
    workflow.update(_video_out_nodes("17", "console_i2v_hq" if hq else "console_i2v", upscale=upscale))
    return workflow


def _build_s2v_workflow(
    image_filename: str,
    audio_filename: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    quality: str = "fast",
    upscale: bool = False,
    clip_name: str | None = None,
) -> dict[str, Any]:
    s2v_neg = (
        "closed mouth, silent, frozen face, still image, no lip movement, static, no talking, "
        "cartoon, extra people, extra limbs, ghosting, double chin, morphing, blurry, text, watermark"
    )
    neg = (negative or "").strip()
    neg = f"{neg}, {s2v_neg}" if neg else s2v_neg
    hq = quality == "hq"
    # 官方标准 S2V：20 步 CFG6。低步数/低 CFG 会石头脸、嘴不动。
    steps = 24 if hq else 20
    cfg = 6.0
    clip_name = clip_name or S2V_CLIP
    workflow: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": S2V_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 5.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}},
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
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["11", 2],
                "denoise": 1.0,
            },
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["6", 0]},
        },
    }
    workflow.update(_video_out_nodes("13", "console_s2v_hq" if hq else "console_s2v", audio_node="8", upscale=upscale))
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


def _build_t2v_workflow(
    prompt: str,
    negative: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    quality: str = "fast",
    upscale: bool = False,
    unet_high: str | None = None,
    unet_low: str | None = None,
    clip_name: str | None = None,
) -> dict[str, Any]:
    """Wan 2.2 T2V 14B: empty latent + text only. No image, no CLIP Vision, no face."""
    neg = negative or WAN_NEG
    hq = quality == "hq"
    unet_high = unet_high or T2V_UNET
    unet_low = unet_low or T2V_UNET_LOW
    clip_name = clip_name or I2V_CLIP
    if hq:
        high_model: list[Any] = ["2", 0]
        low_model: list[Any] = ["4", 0]
        steps = HQ_I2V_STEPS
        switch = HQ_I2V_SWITCH
        cfg = HQ_CFG
        workflow: dict[str, Any] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_high, "weight_dtype": "default"}},
            "2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
            "3": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_low, "weight_dtype": "default"}},
            "4": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": 8.0}},
        }
    else:
        high_model = ["3", 0]
        low_model = ["6", 0]
        steps = 4
        switch = 2
        cfg = 1.0
        workflow = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_high, "weight_dtype": "default"}},
            "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": T2V_LORA_HIGH, "strength_model": 1.0}},
            "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 8.0}},
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_low, "weight_dtype": "default"}},
            "5": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["4", 0], "lora_name": T2V_LORA_LOW, "strength_model": 1.0}},
            "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["5", 0], "shift": 8.0}},
        }
    workflow.update(
        {
            "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": neg}},
            "10": {"class_type": "VAELoader", "inputs": {"vae_name": I2V_VAE}},
            "11": {
                "class_type": "EmptyHunyuanLatentVideo",
                "inputs": {"width": width, "height": height, "length": length, "batch_size": 1},
            },
            "15": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": high_model,
                    "add_noise": "enable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["11", 0],
                    "start_at_step": 0,
                    "end_at_step": switch,
                    "return_with_leftover_noise": "enable",
                },
            },
            "16": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": low_model,
                    "add_noise": "disable",
                    "noise_seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["8", 0],
                    "negative": ["9", 0],
                    "latent_image": ["15", 0],
                    "start_at_step": switch,
                    "end_at_step": steps,
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
        }
    )
    workflow.update(_video_out_nodes("17", "console_t2v_hq" if hq else "console_t2v", upscale=upscale))
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
    quality: str = "fast",
    max_len: int | None = None,
    log_callback: LogCb | None = None,
) -> Path:
    """Run Wan I2V or S2V in ComfyUI. Returns path of raw Comfy output video."""
    image_path = UPLOAD_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Shot image not found: {image_path}")
    image_filename = _copy_to_input(image_path)
    hq = quality == "hq"
    vram_gb = float(_gpu_info()["vram_gb"])
    if hq:
        profile = _select_profile(vram_gb)
    else:
        # fp8 + 4步；大卡仍按 1088×1920 采，吃掉 Flux 底图像素
        high_res = vram_gb >= 45
        profile = {
            "tier": "fast",
            "label": "fp8 4步 LoRA",
            "max_short": 1080 if high_res else 720,
            "i2v_unet": I2V_UNET,
            "i2v_unet_low": I2V_UNET_LOW,
            "clip": I2V_CLIP,
            "native_1080": high_res,
        }
    gen_w, gen_h = _gen_size(width, height, max_short=int(profile["max_short"]))
    upscale = hq and (not profile["native_1080"]) and max(width, height) > max(gen_w, gen_h)
    length = _frame_length(duration_s, max_len=max_len or 49)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    prompt = positive_prompt or "a person moving naturally, cinematic motion"
    use_s2v = mode == "s2v" and bool(audio_name)
    if use_s2v:
        audio_path = UPLOAD_DIR / audio_name
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        audio_filename = _copy_to_input(audio_path)
        workflow = _build_s2v_workflow(
            image_filename, audio_filename, prompt, negative_prompt, gen_w, gen_h, length, seed,
            quality=quality, upscale=upscale, clip_name=S2V_CLIP,
        )
        _log(
            log_callback,
            "info",
            "提交 ComfyUI S2V 工作流",
            {
                "size": f"{gen_w}x{gen_h}",
                "length": length,
                "steps": 24 if hq else 20,
                "cfg": 6.0,
                "shift": 5.0,
                "lora": False,
                "quality": quality,
                "hq_tier": profile["tier"],
                "hq_label": profile["label"],
                "upscale": upscale,
                "context_windows": length > 81,
                "vae_temporal_overlap": 8,
            },
        )
    else:
        workflow = _build_i2v_workflow(
            image_filename, prompt, negative_prompt, gen_w, gen_h, length, seed,
            quality=quality, upscale=upscale,
            unet_high=str(profile["i2v_unet"]), unet_low=str(profile["i2v_unet_low"]),
            clip_name=str(profile["clip"]),
        )
        _log(
            log_callback,
            "info",
            "提交 ComfyUI I2V 工作流",
            {
                "size": f"{gen_w}x{gen_h}",
                "length": length,
                "unet": profile["i2v_unet"],
                "unet_low": profile["i2v_unet_low"],
                "quality": quality,
                "hq_tier": profile["tier"],
                "hq_label": profile["label"],
                "steps": HQ_I2V_STEPS if hq else 4,
                "lora": not hq,
                "upscale": upscale,
            },
        )

    with GPU.hold(log_callback, kind="video"):
        r = httpx.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(f"ComfyUI rejected prompt: {body}")
        prompt_id = body["prompt_id"]
        _log(log_callback, "info", "ComfyUI 视频任务已提交", {"prompt_id": prompt_id, "mode": "s2v" if use_s2v else "i2v", "quality": quality})
        # S2V 固定 20 步；32GB 卡会 partial load，15 分钟经常不够。
        wait_s = 5400 if hq else (3600 if use_s2v else 900)
        return _wait_video(prompt_id, timeout=wait_s, log_callback=log_callback)


def generate_t2v_video(
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    duration_s: float,
    seed: int | None = None,
    quality: str = "fast",
    audio_name: str | None = None,
    log_callback: LogCb | None = None,
) -> Path:
    """Pure Wan 2.2 T2V. No start image, no face swap, no CLIP Vision."""
    hq = quality == "hq"
    info = _gpu_info()
    vram_gb = float(info["vram_gb"])
    if hq:
        profile = _select_t2v_profile(vram_gb)
    else:
        high_res = vram_gb >= 45
        profile = {
            "tier": "fast",
            "label": "T2V fp8 4步 LoRA",
            "max_short": 1080 if high_res else 720,
            "unet_high": T2V_UNET,
            "unet_low": T2V_UNET_LOW,
            "clip": I2V_CLIP,
            "native_1080": high_res,
        }
    gen_w, gen_h = _gen_size(width, height, max_short=int(profile["max_short"]))
    upscale = hq and (not profile["native_1080"]) and max(width, height) > max(gen_w, gen_h)
    length = _frame_length(duration_s, max_len=81)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    prompt = (positive_prompt or "").strip()
    if not prompt:
        raise ValueError("T2V 需要正向提示词")
    workflow = _build_t2v_workflow(
        prompt,
        negative_prompt,
        gen_w,
        gen_h,
        length,
        seed,
        quality=quality,
        upscale=upscale,
        unet_high=str(profile["unet_high"]),
        unet_low=str(profile["unet_low"]),
        clip_name=str(profile["clip"]),
    )
    _log(
        log_callback,
        "info",
        "提交 ComfyUI T2V 工作流",
        {
            "size": f"{gen_w}x{gen_h}",
            "length": length,
            "duration_s": duration_s,
            "unet": profile["unet_high"],
            "unet_low": profile["unet_low"],
            "clip": profile["clip"],
            "quality": quality,
            "hq_tier": profile["tier"],
            "hq_label": profile["label"],
            "steps": HQ_I2V_STEPS if hq else 4,
            "lora": not hq,
            "upscale": upscale,
            "seed": seed,
            "image": False,
            "face": False,
        },
    )
    with GPU.hold(log_callback, kind="video"):
        r = httpx.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=60)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(f"ComfyUI rejected prompt: {body}")
        prompt_id = body["prompt_id"]
        _log(log_callback, "info", "ComfyUI T2V 任务已提交", {"prompt_id": prompt_id, "quality": quality})
        raw = _wait_video(prompt_id, timeout=5400 if hq else 900, log_callback=log_callback)
        out_name = f"t2v_{gen_w}x{gen_h}_{seed}.mp4"
        out_path = OUTPUT_DIR / out_name
        finalize_video(raw, out_path, width, height, audio_name, duration_s)
        CONSOLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        local = CONSOLE_OUTPUT_DIR / out_name
        if local.resolve() != out_path.resolve():
            shutil.copy2(out_path, local)
        _log(log_callback, "info", "T2V 成片已保存", {"path": str(out_path), "local": str(local)})
        return out_path


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
