# coding=utf-8
"""Fast 2-shot 14B fp16 I2V on RTX PRO 6000: forest + bed, 2s each."""
from __future__ import annotations

import asyncio
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMFY_URL, INPUT_DIR, OUTPUT_DIR, SD_CKPT, UPLOAD_DIR
from services import comfy_image, comfy_video

NEG_IMG = (
    "cartoon, anime, 3d render, extra people, crowd, extra limbs, extra arms, extra heads, "
    "deformed genitals, mutated, lowres, blurry, text, watermark, logo, split screen, collage, "
    "ghosting, double exposure"
)
NEG_VID = (
    comfy_video.WAN_NEG
    + ", extra people, extra limbs, morphing bodies, ghosting, double exposure, motion smear, "
    "duplicate couple, extra penis, extra heads, low resolution, still image"
)

SHOTS = [
    {
        "id": "forest",
        "img": (
            "photorealistic cinematic still, vertical 9:16, one adult Chinese man and one adult Chinese woman "
            "having passionate sex in a dense green forest, woman lying on moss looking up moaning, man on top, "
            "naked sweaty bodies, dappled sunlight through leaves, shallow depth of field, sharp skin detail, "
            "two people only, realistic"
        ),
        "vid": (
            "Vertical 9:16 live-action, the couple continues having sex in the forest, man thrusting, "
            "woman moaning with open mouth, natural rhythmic hip movement, leaves slightly moving, "
            "handheld micro shake, sharp subjects, no morphing, no ghosting, two people only"
        ),
        "line": "啊…许总…吴总…快点啊…嗯啊…再快点…",
    },
    {
        "id": "bed",
        "img": (
            "photorealistic cinematic still, vertical 9:16, one adult Chinese man and one adult Chinese woman "
            "having passionate sex on a rumpled white hotel bed, woman on her back clutching sheets moaning, "
            "man on top, warm bedside lamp, naked sweaty bodies, two people only, sharp skin, realistic"
        ),
        "vid": (
            "Vertical 9:16 live-action, the couple continues having sex on the bed, rhythmic thrusting, "
            "woman moaning calling out, sheets wrinkling, warm lamp light, sharp faces, "
            "no morphing, no ghosting, two people only"
        ),
        "line": "许总…吴总…快点啊…啊…用力…嗯啊…",
    },
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_image(prompt_id: str, timeout: int = 180) -> Path:
    name = comfy_image._wait_for_output(prompt_id, timeout=timeout)
    src = OUTPUT_DIR / name
    if src.exists():
        return src
    cands = sorted(OUTPUT_DIR.glob("s2v_ref_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(name)
    return cands[0]


def gen_still(prompt: str, shot_id: str) -> str:
    seed = random.randint(0, 2**32 - 1)
    wf = comfy_image._build_t2i_workflow(prompt, NEG_IMG, width=576, height=1024, seed=seed)
    r = httpx.post(f"{COMFY_URL}/prompt", json=wf, timeout=30)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body)
    pid = body["prompt_id"]
    log(f"{shot_id} 参考图已提交 {pid}")
    src = wait_image(pid)
    name = f"{shot_id}_{random.randint(100000, 999999)}.png"
    dst = UPLOAD_DIR / name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").resize((1088, 1920), Image.Resampling.LANCZOS).save(dst, "PNG")
    log(f"{shot_id} 参考图就绪 {dst} ({dst.stat().st_size} bytes)")
    return name


async def make_voice(text: str, out: Path) -> None:
    import edge_tts

    comm = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural", rate="+8%", pitch="+6Hz")
    await comm.save(str(out))


def make_audio(text: str, shot_id: str) -> str:
    wav = UPLOAD_DIR / f"{shot_id}_voice_{random.randint(1000, 9999)}.mp3"
    asyncio.run(make_voice(text, wav))
    # pad/trim to ~2.2s
    out = UPLOAD_DIR / f"{shot_id}_voice_{random.randint(1000, 9999)}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(wav), "-af", "apad=pad_dur=2.2", "-t", "2.2",
            "-ar", "44100", "-ac", "1", str(out),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log(f"{shot_id} 配音就绪 {out.name}")
    return out.name


def main() -> None:
    log(f"GPU profile {comfy_video.gpu_status()}")
    finals: list[Path] = []
    for shot in SHOTS:
        img_name = gen_still(shot["img"], shot["id"])
        try:
            audio_name = make_audio(shot["line"], shot["id"])
        except Exception as e:
            log(f"{shot['id']} 配音失败，无声出片: {e}")
            audio_name = None
        log(f"{shot['id']} 开始 14B fp16 I2V 2s")
        raw = comfy_video.generate_shot_video(
            image_name=img_name,
            audio_name=audio_name,
            positive_prompt=shot["vid"],
            negative_prompt=NEG_VID,
            width=1088,
            height=1920,
            duration_s=2.0,
            mode="i2v",
            quality="hq",
            log_callback=lambda level, msg, data=None: log(f"{level} {msg} {data or ''}"),
        )
        out = OUTPUT_DIR / f"nsfw_{shot['id']}_fp16.mp4"
        comfy_video.finalize_video(raw, out, 1088, 1920, audio_name, 2.0)
        finals.append(out)
        log(f"{shot['id']} 成片 {out}")
    log("全部完成:")
    for p in finals:
        log(str(p))


if __name__ == "__main__":
    main()
