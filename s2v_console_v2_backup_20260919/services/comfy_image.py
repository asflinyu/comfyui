# coding=utf-8
"""ComfyUI image workflows: T2I / I2I across Flux, Qwen, SDXL, SD1.5, SD3.5."""
from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from services.comfy_runtime import GPU
from config import (
    COMFY_URL,
    FLUX_CLIP_L,
    FLUX_HEIGHT,
    FLUX_T5,
    FLUX_UNET,
    FLUX_VAE,
    FLUX_WIDTH,
    INPUT_DIR,
    OUTPUT_DIR,
    SD_CKPT,
    UPLOAD_DIR,
)

LogCb = Any

FLUX_SCHNELL_UNET = "FLUX1/flux1-schnell-fp8-e4m3fn.safetensors"
FLUX_KONTEXT_UNET = "flux1-dev-kontext_fp8_scaled.safetensors"
QWEN_UNET = "qwen_image_fp8_e4m3fn.safetensors"
QWEN_EDIT_UNET = "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
QWEN_CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_T2I_LORA = "Qwen-Image-Lightning-4steps-V1.0.safetensors"
QWEN_EDIT_LORA = "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
SDXL_CKPT = "sd_xl_base_1.0.safetensors"
SD35_CKPT = "sd3.5_large_fp8_scaled.safetensors"
SD3_CLIP_L = "clip_l.safetensors"
SD3_CLIP_G = "clip_g.safetensors"
SD3_T5 = "t5xxl_fp8_e4m3fn_scaled.safetensors"
UPSCALE_MODEL = "4x-UltraSharp.pth"


def _seed(seed: int | None) -> int:
    return random.randint(0, 2**32 - 1) if seed is None else int(seed)


def _align(n: int, step: int = 16) -> int:
    return max(step, int(n) // step * step)


def _wrap(nodes: dict[str, Any]) -> dict[str, Any]:
    return {"prompt": nodes}


def _ksampler(
    model: list[Any],
    positive: list[Any],
    negative: list[Any],
    latent: list[Any],
    seed: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    denoise: float,
) -> dict[str, Any]:
    return {
        "class_type": "KSampler",
        "inputs": {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent_image": latent,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": denoise,
        },
    }


# --- workflow builders (real Comfy API graphs) ---

def _build_flux_t2i(prompt: str, negative: str, width: int, height: int, seed: int, unet: str, steps: int, prefix: str) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["1", 0], "max_shift": 1.15, "base_shift": 0.5, "width": width, "height": height},
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative or ""}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 3.5}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": _ksampler(["2", 0], ["6", 0], ["5", 0], ["8", 0], seed, steps, 1.0, "euler", "simple", 1.0),
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["7", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
    })


def _build_flux_kontext_i2i(prompt: str, negative: str, image_filename: str, width: int, height: int, seed: int) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": FLUX_KONTEXT_UNET, "weight_dtype": "default"}},
        "2": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["1", 0], "max_shift": 1.15, "base_shift": 0.5, "width": width, "height": height},
        },
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative or ""}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 3.5}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "8": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "9": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["8", 0]}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["7", 0]}},
        "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["10", 0]}},
        "12": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "13": _ksampler(["2", 0], ["11", 0], ["5", 0], ["12", 0], seed, 20, 1.0, "euler", "simple", 1.0),
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["7", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"images": ["14", 0], "filename_prefix": "img_flux_kontext"}},
    })


def _build_qwen_t2i(prompt: str, negative: str, width: int, height: int, seed: int) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": QWEN_T2I_LORA, "strength_model": 1.0},
        },
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": negative or " "}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": _ksampler(["3", 0], ["5", 0], ["6", 0], ["8", 0], seed, 4, 1.0, "euler", "simple", 1.0),
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["7", 0]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "img_qwen_t2i"}},
    })


def _build_qwen_edit_i2i(prompt: str, negative: str, image_filename: str, width: int, height: int, seed: int) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_EDIT_UNET, "weight_dtype": "default"}},
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": QWEN_EDIT_LORA, "strength_model": 1.0},
        },
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3.1}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "6": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "7": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {"clip": ["4", 0], "prompt": prompt, "vae": ["5", 0], "image": ["6", 0]},
        },
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": negative or " "}},
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": _ksampler(["3", 0], ["7", 0], ["8", 0], ["9", 0], seed, 4, 1.0, "euler", "simple", 1.0),
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["5", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "img_qwen_edit"}},
    })


def _build_ckpt_t2i(ckpt: str, prompt: str, negative: str, width: int, height: int, seed: int, steps: int, cfg: float, sampler: str, scheduler: str, prefix: str, sd3_latent: bool = False) -> dict[str, Any]:
    latent_node = (
        {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
        if sd3_latent
        else {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    )
    nodes: dict[str, Any] = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative or ""}},
        "4": latent_node,
        "5": _ksampler(["1", 0], ["2", 0], ["3", 0], ["4", 0], seed, steps, cfg, sampler, scheduler, 1.0),
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }
    if sd3_latent:
        nodes["8"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 3.0}}
        nodes["5"]["inputs"]["model"] = ["8", 0]
    return _wrap(nodes)


def _build_ckpt_i2i(ckpt: str, prompt: str, negative: str, image_filename: str, seed: int, steps: int, cfg: float, denoise: float, sampler: str, scheduler: str, prefix: str) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": prompt}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative or ""}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": _ksampler(["1", 0], ["2", 0], ["3", 0], ["5", 0], seed, steps, cfg, sampler, scheduler, denoise),
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    })


def _build_upscale(image_filename: str) -> dict[str, Any]:
    return _wrap({
        "1": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["1", 0], "image": ["2", 0]}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "img_upscale"}},
    })


IMAGE_WORKFLOWS: dict[str, dict[str, Any]] = {
    "flux_dev": {
        "id": "flux_dev",
        "label": "Flux Dev 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": FLUX_WIDTH,
        "height": FLUX_HEIGHT,
        "timeout": 300,
        "prefix": "s2v_flux",
        "desc": "Flux Dev fp8，竖屏 1280×2304，最强静帧",
    },
    "flux_schnell": {
        "id": "flux_schnell",
        "label": "Flux Schnell 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": 1024,
        "height": 1536,
        "timeout": 180,
        "prefix": "img_flux_schnell",
        "desc": "Flux Schnell 4 步，更快出图",
    },
    "qwen_t2i": {
        "id": "qwen_t2i",
        "label": "Qwen 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": 1024,
        "height": 1536,
        "timeout": 180,
        "prefix": "img_qwen_t2i",
        "desc": "Qwen-Image fp8 + Lightning 4 步，中文提示更好",
    },
    "sdxl_t2i": {
        "id": "sdxl_t2i",
        "label": "SDXL 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": 832,
        "height": 1216,
        "timeout": 180,
        "prefix": "img_sdxl_t2i",
        "desc": "SDXL 1.0 Base，1024 级竖屏",
    },
    "sd15_t2i": {
        "id": "sd15_t2i",
        "label": "SD1.5 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": 576,
        "height": 1024,
        "timeout": 120,
        "prefix": "s2v_ref",
        "desc": "majicMIX Realistic v7，出图快",
    },
    "sd35_t2i": {
        "id": "sd35_t2i",
        "label": "SD3.5 文生图",
        "kind": "t2i",
        "needs_image": False,
        "width": 1024,
        "height": 1536,
        "timeout": 300,
        "prefix": "img_sd35_t2i",
        "desc": "SD3.5 Large fp8",
    },
    "flux_kontext": {
        "id": "flux_kontext",
        "label": "Flux Kontext 图生图",
        "kind": "i2i",
        "needs_image": True,
        "width": 832,
        "height": 1248,
        "timeout": 300,
        "prefix": "img_flux_kontext",
        "desc": "按文字改图，尽量保住构图和人物",
    },
    "qwen_edit": {
        "id": "qwen_edit",
        "label": "Qwen Edit 图生图",
        "kind": "i2i",
        "needs_image": True,
        "width": 1024,
        "height": 1536,
        "timeout": 180,
        "prefix": "img_qwen_edit",
        "desc": "Qwen-Image-Edit 2509，中文改图指令",
    },
    "sdxl_i2i": {
        "id": "sdxl_i2i",
        "label": "SDXL 图生图",
        "kind": "i2i",
        "needs_image": True,
        "width": 832,
        "height": 1216,
        "timeout": 180,
        "prefix": "img_sdxl_i2i",
        "desc": "SDXL 重绘底图，可调 denoising",
    },
    "sd15_i2i": {
        "id": "sd15_i2i",
        "label": "SD1.5 图生图",
        "kind": "i2i",
        "needs_image": True,
        "width": 576,
        "height": 1024,
        "timeout": 120,
        "prefix": "s2v_ref_i2i",
        "desc": "majicMIX 重绘，速度快",
    },
    "upscale": {
        "id": "upscale",
        "label": "4x UltraSharp 放大",
        "kind": "i2i",
        "needs_image": True,
        "width": 0,
        "height": 0,
        "timeout": 90,
        "prefix": "img_upscale",
        "desc": "Real-ESRGAN / UltraSharp 把已有图放大，不改内容",
    },
}


def list_workflows() -> list[dict[str, Any]]:
    order = [
        "flux_dev", "qwen_t2i", "sdxl_t2i", "flux_schnell", "sd15_t2i", "sd35_t2i",
        "flux_kontext", "qwen_edit", "sdxl_i2i", "sd15_i2i", "upscale",
    ]
    return [IMAGE_WORKFLOWS[k] for k in order if k in IMAGE_WORKFLOWS]


def _flux_native_size(width: int, height: int) -> tuple[int, int]:
    if height >= width:
        return FLUX_WIDTH, FLUX_HEIGHT
    return FLUX_HEIGHT, FLUX_WIDTH


def _sd15_native_size(width: int, height: int) -> tuple[int, int]:
    if height >= width:
        return 576, 1024
    return 1024, 576


def _upscale_to(src: Path, dst: Path, width: int, height: int) -> None:
    from PIL import Image

    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize((width, height), Image.Resampling.LANCZOS)
        im.save(dst, format="PNG")


def _prepare_input_image(image_name: str) -> str:
    src = UPLOAD_DIR / image_name
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")
    dest = INPUT_DIR / image_name
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
    return image_name


def _wait_for_output(prompt_id: str, timeout: int = 120, log_callback: LogCb = None) -> str:
    def _log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        if callable(log_callback):
            log_callback(level, msg, data)

    deadline = time.time() + timeout
    poll_count = 0
    _log("info", "开始轮询 ComfyUI history", {"prompt_id": prompt_id, "timeout": timeout})
    while time.time() < deadline:
        poll_count += 1
        r = httpx.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        entry = data.get(prompt_id, {})
        outputs = entry.get("outputs", {})
        for node_out in outputs.values():
            for img in node_out.get("images", []) or []:
                filename = img.get("filename") if isinstance(img, dict) else img
                if filename:
                    _log("info", "ComfyUI history 返回输出", {"prompt_id": prompt_id, "poll_count": poll_count, "filename": filename})
                    return filename
        status = entry.get("status", {})
        if status.get("status_msg") == "error":
            _log("error", "ComfyUI 生成失败", {"prompt_id": prompt_id, "status": status})
            raise RuntimeError(f"ComfyUI prompt failed: {status}")
        time.sleep(1)
    raise TimeoutError(f"ComfyUI generation timed out after {timeout}s")


def _build_named_workflow(
    workflow_id: str,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    image_filename: str,
    denoise: float,
) -> tuple[dict[str, Any], str]:
    meta = IMAGE_WORKFLOWS[workflow_id]
    prefix = str(meta["prefix"])
    if workflow_id == "flux_dev":
        return _build_flux_t2i(prompt, negative, width, height, seed, FLUX_UNET, 20, prefix), prefix
    if workflow_id == "flux_schnell":
        return _build_flux_t2i(prompt, negative, width, height, seed, FLUX_SCHNELL_UNET, 4, prefix), prefix
    if workflow_id == "flux_kontext":
        return _build_flux_kontext_i2i(prompt, negative, image_filename, width, height, seed), prefix
    if workflow_id == "qwen_t2i":
        return _build_qwen_t2i(prompt, negative, width, height, seed), prefix
    if workflow_id == "qwen_edit":
        return _build_qwen_edit_i2i(prompt, negative, image_filename, width, height, seed), prefix
    if workflow_id == "sdxl_t2i":
        return _build_ckpt_t2i(SDXL_CKPT, prompt, negative, width, height, seed, 28, 6.0, "dpmpp_2m", "karras", prefix), prefix
    if workflow_id == "sdxl_i2i":
        return _build_ckpt_i2i(SDXL_CKPT, prompt, negative, image_filename, seed, 24, 6.0, denoise, "dpmpp_2m", "karras", prefix), prefix
    if workflow_id == "sd15_t2i":
        return _build_ckpt_t2i(SD_CKPT, prompt, negative, width, height, seed, 22, 7.0, "euler", "normal", prefix), prefix
    if workflow_id == "sd15_i2i":
        return _build_ckpt_i2i(SD_CKPT, prompt, negative, image_filename, seed, 20, 7.0, denoise, "euler", "normal", prefix), prefix
    if workflow_id == "sd35_t2i":
        return _build_ckpt_t2i(SD35_CKPT, prompt, negative, width, height, seed, 28, 4.5, "euler", "sgm_uniform", prefix, sd3_latent=True), prefix
    if workflow_id == "upscale":
        return _build_upscale(image_filename), prefix
    raise ValueError(f"未知生图工作流: {workflow_id}")


def generate_image(
    prompt: str,
    negative: str = "",
    width: int = 256,
    height: int = 256,
    seed: int | None = None,
    filename_prefix: str = "console_ref",
    prompt_en: str = "",
    source_image_name: str = "",
    denoise: float = 0.65,
    workflow: str = "",
    log_callback: Any = None,
    count: int = 1,
) -> str:
    """Run a named T2I/I2I workflow and save into UPLOAD_DIR. Returns the first image name."""
    names = generate_images(
        prompt,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        filename_prefix=filename_prefix,
        prompt_en=prompt_en,
        source_image_name=source_image_name,
        denoise=denoise,
        workflow=workflow,
        log_callback=log_callback,
        count=count,
    )
    return names[0]


def generate_images(
    prompt: str,
    negative: str = "",
    width: int = 256,
    height: int = 256,
    seed: int | None = None,
    filename_prefix: str = "console_ref",
    prompt_en: str = "",
    source_image_name: str = "",
    denoise: float = 0.65,
    workflow: str = "",
    log_callback: Any = None,
    count: int = 1,
) -> list[str]:
    """Generate 1–4 images. Same GPU hold, seeds 递增以免撞图。"""

    def _log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        if callable(log_callback):
            log_callback(level, msg, data)

    gen_prompt = prompt_en or prompt or ""
    wf_id = (workflow or "").strip()
    if not wf_id:
        wf_id = "sd15_i2i" if source_image_name else "flux_dev"
        # Keep old storyboard behavior: English-biased prompt injection.
        if not gen_prompt and wf_id != "upscale":
            raise ValueError("No prompt provided for image generation")
        if wf_id != "upscale":
            low = gen_prompt.lower()
            two_people = any(k in low for k in ("two ", "couple", "man and", "woman and", "a man", "a woman and"))
            if "photorealistic" not in low:
                gen_prompt = f"photorealistic, {gen_prompt}"
            if not two_people and "one woman" not in low and "one man" not in low:
                gen_prompt = f"solo, one person only, {gen_prompt}"
            animal_neg = (
                "cat, kitten, dog, puppy, animal, cartoon, anime, extra limbs, deformed face, "
                "split screen, collage, stacked bodies"
            )
            if not two_people:
                animal_neg += ", two people, twins, duplicate, extra person, extra arms"
            if animal_neg not in (negative or ""):
                negative = f"{negative}, {animal_neg}" if negative else animal_neg
    elif wf_id not in IMAGE_WORKFLOWS:
        raise ValueError(f"未知生图工作流: {wf_id}")

    meta = IMAGE_WORKFLOWS[wf_id]
    if meta["needs_image"] and not source_image_name:
        raise ValueError(f"{meta['label']} 需要先上传一张底图")
    if wf_id != "upscale" and not gen_prompt:
        raise ValueError("请填写提示词")

    n_out = 1 if wf_id == "upscale" else max(1, min(4, int(count or 1)))
    native_w = _align(int(width or meta["width"] or 1024))
    native_h = _align(int(height or meta["height"] or 1024))
    if wf_id == "flux_dev":
        native_w, native_h = _flux_native_size(native_w, native_h)
    if wf_id in {"sd15_t2i", "sd15_i2i"}:
        native_w, native_h = _sd15_native_size(native_w, native_h)

    image_filename = ""
    if source_image_name:
        image_filename = _prepare_input_image(source_image_name)

    timeout = int(meta["timeout"])
    names: list[str] = []
    _log("info", f"提交生图工作流 {wf_id}", {
        "workflow": wf_id,
        "size": f"{native_w}x{native_h}",
        "count": n_out,
        "has_image": bool(image_filename),
        "prompt": gen_prompt[:120],
    })
    with GPU.hold(log_callback, kind="image"):
        for i in range(n_out):
            seed_i = _seed(None if seed is None else int(seed) + i)
            graph, prefix = _build_named_workflow(
                wf_id, gen_prompt, negative, native_w, native_h, seed_i, image_filename, float(denoise),
            )
            if n_out > 1:
                _log("info", f"正在生成第 {i + 1}/{n_out} 张", {"seed": seed_i})
            r = httpx.post(f"{COMFY_URL}/prompt", json=graph, timeout=30)
            r.raise_for_status()
            prompt_id = r.json()["prompt_id"]
            _log("info", "ComfyUI prompt 已提交", {"prompt_id": prompt_id, "workflow": wf_id, "index": i + 1})

            output_filename = _wait_for_output(prompt_id, timeout=timeout, log_callback=log_callback)
            src = OUTPUT_DIR / output_filename
            if not src.exists():
                candidates = sorted(
                    OUTPUT_DIR.glob(f"{prefix}_*.{Path(output_filename).suffix or 'png'}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not candidates:
                    raise RuntimeError(f"ComfyUI output not found: {output_filename}")
                src = candidates[0]

            suffix = src.suffix or ".png"
            new_name = f"{filename_prefix}_{random.randint(100000, 999999)}{suffix}"
            dst = UPLOAD_DIR / new_name
            shutil.copy2(src, dst)
            names.append(new_name)
            _log("info", "参考图生成完成", {"image_name": new_name, "workflow": wf_id, "index": i + 1, "src": str(src)})
    return names


def _normalize_negative(shot_neg: str) -> str:
    defaults = "text, watermark, signature, logo, low quality, blurry, distorted face, extra limbs, deformed hands"
    if not shot_neg:
        return defaults
    return f"{shot_neg}, {defaults}" if defaults not in shot_neg else shot_neg
