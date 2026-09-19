# coding=utf-8
"""S2V 分镜控制台 V2 配置（自包含，避免与 s2v_console 原目录冲突）。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
INPUT_DIR = COMFY_ROOT / "input"
OUTPUT_DIR = COMFY_ROOT / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
if DEEPSEEK_API_KEY.startswith("sk-请") or DEEPSEEK_API_KEY in {"YOUR_KEY", "sk-xxx", "changeme"}:
    DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# Moonshot Kimi (OpenAI-compatible) — preferred for long copywriting
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY", "").strip()
if MOONSHOT_API_KEY.startswith("sk-请") or MOONSHOT_API_KEY in {"YOUR_KEY", "sk-xxx", "changeme"}:
    MOONSHOT_API_KEY = ""
MOONSHOT_BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
MOONSHOT_MODEL = os.environ.get("MOONSHOT_MODEL", "kimi-k2.7-code-highspeed")

LLM_API_KEY = MOONSHOT_API_KEY or DEEPSEEK_API_KEY
LLM_BASE_URL = MOONSHOT_BASE_URL if MOONSHOT_API_KEY else DEEPSEEK_BASE_URL
LLM_MODEL = MOONSHOT_MODEL if MOONSHOT_API_KEY else DEEPSEEK_MODEL

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:6006")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:6009")
CONSOLE_HOST = os.environ.get("CONSOLE_HOST", "0.0.0.0")
CONSOLE_PORT = int(os.environ.get("CONSOLE_PORT", "6010"))

FPS = 16
MIN_SHOTS = 2
MAX_SHOTS = 10
MAX_VIDEOS = 4

RESOLUTION_PRESETS = {
    "smoke_128": {"width": 128, "height": 128, "label": "冒烟 128×128"},
    "smoke_256": {"width": 256, "height": 256, "label": "试跑 256×256"},
    "draft_480": {"width": 480, "height": 832, "label": "草稿 480×832"},
    "final_720": {"width": 720, "height": 1280, "label": "正式 720×1280"},
    "vertical_854": {"width": 854, "height": 1536, "label": "竖屏 854×1536"},
    "vertical_1080": {"width": 1080, "height": 1920, "label": "竖屏 1080×1920"},
    "flux_1280": {"width": 1280, "height": 2304, "label": "Flux 竖屏 1280×2304"},
}

# fast = fp8 + 4步 LoRA（推荐）；hq = fp16 20步，很慢
QUALITY_PRESETS = {
    "fast": {"label": "加速 14B（fp8 + 4步 LoRA）", "steps": 4},
    "hq": {"label": "高清 14B（fp16 20步，很慢）", "steps": 20},
}
DEFAULT_QUALITY = "fast"
UPSCALE_MODEL = "RealESRGAN_x2.pth"

S2V_UNET = "wan2.2_s2v_14B_fp8_scaled.safetensors"
S2V_LORA = "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
S2V_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
S2V_VAE = "wan_2.1_vae.safetensors"
S2V_AUDIO_ENC = "wav2vec2_large_english_fp16.safetensors"

I2V_UNET = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
I2V_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
I2V_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
I2V_VAE = "wan_2.1_vae.safetensors"

SD_CKPT = "majicMIX_realistic_v7.safetensors"

FLUX_UNET = "FLUX1/flux1-dev-fp8.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_T5 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
FLUX_VAE = "ae.safetensors"
FLUX_WIDTH = 1280
FLUX_HEIGHT = 2304

DEFAULT_NEGATIVE = (
    "cat, kitten, dog, puppy, animal, paws, fur, cartoon, anime, 3D render, drawing, illustration, "
    "deformed faces, mutated paws, twisted limbs, shape-shifting, text overlays, glowing fantasy effects, "
    "blurry subject, low resolution, creepy, over-saturated colors, weird artifacts, morphing textures, "
    "sharp background, clear background, distorted UI, extra limbs."
)

DEFAULT_VOICE = "01 温柔女声·播音腔"
