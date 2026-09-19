# coding=utf-8
"""Bed scene 5s, sharp start still (4x ESRGAN) + Wan 14B fp16 native 1088x1920."""
from __future__ import annotations

import asyncio
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMFY_URL, INPUT_DIR, OUTPUT_DIR, UPLOAD_DIR
from services import comfy_image, comfy_video

SRC_STILL = Path("/root/ComfyUI/output/s2v_ref_00053_.png")
OUT_DIR = Path("/root/ComfyUI/s2v_console_v2/output")
NEG_VID = (
    comfy_video.WAN_NEG
    + ", extra people, extra limbs, morphing bodies, ghosting, double exposure, motion smear, "
    "duplicate couple, extra heads, low resolution, still image, blurry, soft focus, jpeg artifacts"
)
VID_PROMPT = (
    "Vertical 9:16 live-action, the couple continues having sex on the bed for several seconds, "
    "natural rhythmic thrusting, woman moaning with open mouth, sheets wrinkling, warm lamp light, "
    "sharp faces and skin texture, no morphing, no ghosting, two people only, crisp details"
)
LINE = "啊…许总…吴总…快点啊…嗯啊…再快点…许总用力…吴总…快点啊…"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sharpen_still() -> str:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local = INPUT_DIR / "bed_src_576.png"
    shutil.copy2(SRC_STILL, local)
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": local.name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}},
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
        },
        "4": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": 1088,
                "height": 1920,
                "crop": "center",
            },
        },
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "bed_sharp"}},
    }
    r = httpx.post(f"{COMFY_URL}/prompt", json={"prompt": wf}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body)
    pid = body["prompt_id"]
    log(f"锐化参考图已提交 {pid}")
    name = comfy_image._wait_for_output(pid, timeout=180)
    src = OUTPUT_DIR / name
    if not src.exists():
        cands = sorted(OUTPUT_DIR.glob("bed_sharp_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        src = cands[0]
    out_name = f"bed_sharp_{random.randint(100000, 999999)}.png"
    dst = UPLOAD_DIR / out_name
    shutil.copy2(src, dst)
    log(f"锐化参考图就绪 {dst} {dst.stat().st_size} bytes")
    return out_name


async def _tts(text: str, dest: Path) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural", rate="+6%", pitch="+8Hz")
    await comm.save(str(dest))


def make_audio() -> str:
    mp3 = UPLOAD_DIR / f"bed5s_{random.randint(1000,9999)}.mp3"
    asyncio.run(_tts(LINE, mp3))
    wav = UPLOAD_DIR / f"bed5s_{random.randint(1000,9999)}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3), "-af", "apad=pad_dur=5.2", "-t", "5.2", "-ar", "44100", "-ac", "1", str(wav)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log(f"配音就绪 {wav.name}")
    return wav.name


def main() -> None:
    log(f"GPU {comfy_video.gpu_status()}")
    img = sharpen_still()
    audio = make_audio()
    log("开始 14B fp16 I2V 5s 81帧")
    raw = comfy_video.generate_shot_video(
        image_name=img,
        audio_name=audio,
        positive_prompt=VID_PROMPT,
        negative_prompt=NEG_VID,
        width=1088,
        height=1920,
        duration_s=5.0,
        mode="i2v",
        quality="hq",
        max_len=81,
        log_callback=lambda level, msg, data=None: log(f"{level} {msg} {data or ''}"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "nsfw_bed_5s_fp16.mp4"
    comfy_video.finalize_video(raw, out, 1088, 1920, audio, 5.0)
    shutil.copy2(out, OUTPUT_DIR / "nsfw_bed_5s_fp16.mp4")
    log(f"成片 {out}")


if __name__ == "__main__":
    main()
