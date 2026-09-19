# coding=utf-8
"""MuseTalk 1.5 口型：隐空间重绘嘴部（不拉伸原图嘴唇）。双卡时跑 GPU 1，单卡走 GPU 0。"""
from __future__ import annotations

import json
import math
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from config import OUTPUT_DIR, ROOT, UPLOAD_DIR

LogCb = Callable[[str, str, dict[str, Any] | None], None]
FPS_OUT = 25
MAX_DURATION_S = 45.0

MUSETALK_UNET = Path("/.autodl-model/data/TMElyralab/MuseTalk/musetalkV15/unet.pth")
MUSETALK_CFG = Path("/.autodl-model/data/TMElyralab/MuseTalk/musetalkV15/musetalk.json")
VAE_SAFE = Path("/root/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors")
WHISPER_ID = "/root/autodl-tmp/models/whisper-tiny"
DEVICE = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def _log(cb: LogCb | None, level: str, msg: str, data: dict[str, Any] | None = None) -> None:
    if callable(cb):
        cb(level, msg, data)


def _ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            text=True,
            timeout=15,
        )
        return max(0.0, float((out or "0").strip() or 0))
    except Exception:
        return 0.0


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 384, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(device=x.device, dtype=x.dtype)


class _Bundle:
    def __init__(self) -> None:
        self.ready = False
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.extractor = None
        self.scale = 0.18215
        self.dtype = torch.float32

    def load(self, log: LogCb | None = None) -> None:
        if self.ready:
            return
        if not MUSETALK_UNET.exists():
            raise RuntimeError(f"找不到 MuseTalk 权重: {MUSETALK_UNET}")
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import AutoFeatureExtractor, WhisperModel

        _log(log, "info", "加载 MuseTalk 1.5（第二块 GPU，重绘嘴部）")
        try:
            self.vae = AutoencoderKL.from_single_file(
                str(VAE_SAFE), torch_dtype=self.dtype, local_files_only=True
            )
        except Exception:
            self.vae = AutoencoderKL.from_single_file(str(VAE_SAFE), torch_dtype=self.dtype)
        self.vae.to(self.device).eval()
        self.scale = float(getattr(self.vae.config, "scaling_factor", 0.18215) or 0.18215)

        cfg = json.loads(MUSETALK_CFG.read_text(encoding="utf-8"))
        self.unet = UNet2DConditionModel(**{k: v for k, v in cfg.items() if not str(k).startswith("_")})
        raw = torch.load(str(MUSETALK_UNET), map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if isinstance(raw, dict) and any(k.startswith("model.") for k in raw):
            raw = {k[6:] if k.startswith("model.") else k: v for k, v in raw.items()}
        self.unet.load_state_dict(raw, strict=False)
        self.unet.to(self.device, dtype=self.dtype).eval()

        self.pe = _PositionalEncoding(384).to(self.device, dtype=self.dtype)
        self.extractor = AutoFeatureExtractor.from_pretrained(WHISPER_ID, local_files_only=True)
        self.whisper = WhisperModel.from_pretrained(WHISPER_ID, local_files_only=True).to(self.device, dtype=self.dtype).eval()
        self.whisper.requires_grad_(False)
        self.ready = True
        _log(log, "info", "MuseTalk 已就绪")


_BUNDLE = _Bundle()


# MediaPipe Face Mesh 轮廓，对齐官方 dwpose 68 点脸框
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]
_LIP_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]


def _detect_face(bgr: np.ndarray) -> dict[str, Any]:
    import mediapipe as mp

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
    )
    try:
        res = mesh.process(rgb)
    finally:
        mesh.close()
    if not res.multi_face_landmarks:
        raise RuntimeError("照片里没有检测到人脸，请换一张清晰正脸")
    pts = np.array([[p.x * w, p.y * h] for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
    oval = pts[_FACE_OVAL]
    chin_y = float(pts[152, 1])
    # 多留额头，让嘴部落在 256 裁切下半，避免融合时上唇被原图盖住
    x1i = int(np.clip(oval[:, 0].min() - 4, 0, w - 2))
    x2i = int(np.clip(oval[:, 0].max() + 4, x1i + 8, w))
    y1i = int(np.clip(oval[:, 1].min() - 8, 0, h - 2))
    y2i = int(np.clip(chin_y + 20, y1i + 8, h))
    return {"bbox": (x1i, y1i, x2i, y2i), "pts": pts}


def _face_bbox(bgr: np.ndarray) -> tuple[int, int, int, int]:
    return _detect_face(bgr)["bbox"]


def _to_tensor(bgr256: np.ndarray, half_mask: bool) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr256, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if half_mask:
        rgb[rgb.shape[0] // 2 :, :, :] = 0.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)


@torch.no_grad()
def _encode_unet_input(bgr256: np.ndarray, bundle: _Bundle) -> torch.Tensor:
    masked = _to_tensor(bgr256, True).to(bundle.device, dtype=bundle.dtype)
    full = _to_tensor(bgr256, False).to(bundle.device, dtype=bundle.dtype)
    zm = bundle.vae.encode(masked).latent_dist.sample() * bundle.scale
    zf = bundle.vae.encode(full).latent_dist.sample() * bundle.scale
    return torch.cat([zm, zf], dim=1)


@torch.no_grad()
def _decode(latents: torch.Tensor, bundle: _Bundle) -> list[np.ndarray]:
    img = bundle.vae.decode((1.0 / bundle.scale) * latents).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    return [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in arr]


def _whisper_chunks(wav_path: Path, fps: int, bundle: _Bundle) -> torch.Tensor:
    import librosa

    audio, sr = librosa.load(str(wav_path), sr=16000)
    seg_len = 30 * sr
    feats = []
    for i in range(0, max(1, len(audio)), seg_len):
        chunk = audio[i : i + seg_len]
        if len(chunk) < 160:
            continue
        feat = bundle.extractor(chunk, return_tensors="pt", sampling_rate=16000).input_features
        feats.append(feat.to(bundle.device, dtype=bundle.dtype))
    if not feats:
        raise RuntimeError("音频太短，无法提取口型特征")

    hidden = []
    for feat in feats:
        hs = bundle.whisper.encoder(feat, output_hidden_states=True).hidden_states
        hidden.append(torch.stack(hs, dim=2))
    whisper_feature = torch.cat(hidden, dim=1)
    audio_fps = 50
    num_frames = max(1, math.floor((len(audio) / 16000.0) * fps))
    actual = math.floor((len(audio) / 16000.0) * audio_fps)
    whisper_feature = whisper_feature[:, :actual, ...]
    pad_l, pad_r = 2, 2
    frame_win = 2 * (pad_l + pad_r + 1)
    mult = audio_fps / float(fps)
    pad_n = math.ceil(mult)
    whisper_feature = torch.cat(
        [
            torch.zeros_like(whisper_feature[:, : pad_n * pad_l]),
            whisper_feature,
            torch.zeros_like(whisper_feature[:, : pad_n * 3 * pad_r]),
        ],
        1,
    )
    clips = []
    for fi in range(num_frames):
        ai = math.floor(fi * mult)
        clip = whisper_feature[:, ai : ai + frame_win]
        if clip.shape[1] < frame_win:
            clip = torch.nn.functional.pad(clip, (0, 0, 0, 0, 0, frame_win - clip.shape[1]))
        clips.append(clip)
    prompts = torch.cat(clips, dim=0)
    return rearrange(prompts, "b c h w -> b (c h) w")


def _blend_mouth(
    ori: np.ndarray,
    gen256: np.ndarray,
    bbox: tuple[int, int, int, int],
    pts: np.ndarray | None = None,
) -> np.ndarray:
    """官方 get_image：扩框 1.5、只融下半脸。嘴唇区域强制 100% 用生成图，避免原图闭口盖掉张嘴。"""
    x1, y1, x2, y2 = bbox
    fh, fw = y2 - y1, x2 - x1
    if fh < 8 or fw < 8:
        return ori
    gen = cv2.resize(gen256, (fw, fh), interpolation=cv2.INTER_LANCZOS4)
    lo = gen[fh // 2 :]
    gen[fh // 2 :] = np.clip(
        cv2.addWeighted(lo, 1.35, cv2.GaussianBlur(lo, (0, 0), 1.0), -0.35, 0), 0, 255
    ).astype(np.uint8)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    s = int(max(fw, fh) // 2 * 1.5)
    ih, iw = ori.shape[:2]
    xs, ys = max(0, cx - s), max(0, cy - s)
    xe, ye = min(iw, cx + s), min(ih, cy + s)
    large = ori[ys:ye, xs:xe].copy()
    lh, lw = large.shape[:2]
    mask = np.zeros((lh, lw), np.float32)
    lip_mask = np.zeros((lh, lw), np.float32)
    if pts is not None and len(pts) > 152:
        oval = pts[_FACE_OVAL].copy()
        oval[:, 0] -= xs
        oval[:, 1] -= ys
        cv2.fillPoly(mask, [np.round(oval).astype(np.int32)], 1.0)
        lips = pts[_LIP_OUTER].copy()
        lips[:, 0] -= xs
        lips[:, 1] -= ys
        cv2.fillPoly(lip_mask, [np.round(lips).astype(np.int32)], 1.0)
        kd = max(5, int(0.045 * fw) | 1)
        lip_mask = cv2.dilate(lip_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    else:
        mx1, my1 = x1 - xs, y1 - ys
        mask[max(0, my1):min(lh, y2 - ys), max(0, mx1):min(lw, x2 - xs)] = 1.0
    # 比官方 0.5 更靠上，避免上唇落在羽化带被原图闭口冲淡
    mask[: int(lh * 0.35), :] = 0.0
    boxm = np.zeros_like(mask)
    boxm[max(0, y1 - ys):min(lh, y2 - ys), max(0, x1 - xs):min(lw, x2 - xs)] = 1.0
    mask *= boxm
    k = int(0.05 * lw // 2 * 2) + 1
    k = max(3, k)
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask = np.maximum(mask, lip_mask * boxm)
    layer = large.astype(np.float32)
    px, py = x1 - xs, y1 - ys
    gx0, gy0 = max(0, px), max(0, py)
    gx1, gy1 = min(lw, px + fw), min(lh, py + fh)
    sx0, sy0 = gx0 - px, gy0 - py
    layer[gy0:gy1, gx0:gx1] = gen[sy0:sy0 + (gy1 - gy0), sx0:sx0 + (gx1 - gx0)].astype(np.float32)
    m3 = mask[..., None]
    fused = np.clip(large.astype(np.float32) * (1.0 - m3) + layer * m3, 0, 255).astype(np.uint8)
    out = ori.copy()
    out[ys:ye, xs:xe] = fused
    return out


def _save_mp4(frames: list[np.ndarray], wav: Path, dest: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    w, h = _even(w), _even(h)
    with tempfile.TemporaryDirectory(prefix="muse_") as td:
        raw = Path(td) / "f.raw"
        with raw.open("wb") as fh:
            for fr in frames:
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                fh.write(np.ascontiguousarray(fr).tobytes())
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
            "-i", str(raw),
            "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart",
            str(dest),
        ]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        if run.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {run.stderr[-400:].decode('utf-8', errors='ignore')}")


def _write_outputs(src: Path, prefix: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{random.randint(100000, 999999)}.mp4"
    dest = OUTPUT_DIR / name
    dest.write_bytes(src.read_bytes())
    local = ROOT / "output" / name
    if local.resolve() != dest.resolve():
        local.write_bytes(dest.read_bytes())
    return name


@torch.no_grad()
def _infer_frames(
    src_frames: list[np.ndarray],
    faces: list[dict[str, Any]],
    wav: Path,
    log: LogCb | None,
    batch_size: int = 8,
) -> list[np.ndarray]:
    bundle = _BUNDLE
    bundle.load(log)
    latents = []
    for fr, face in zip(src_frames, faces):
        x1, y1, x2, y2 = face["bbox"]
        crop = cv2.resize(fr[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(_encode_unet_input(crop, bundle))
    chunks = _whisper_chunks(wav, FPS_OUT, bundle)
    n = int(chunks.shape[0])
    cycle_z = latents + latents[::-1]
    cycle_fr = src_frames + src_frames[::-1]
    cycle_face = faces + faces[::-1]
    out: list[np.ndarray] = []
    steps = torch.tensor([0], device=bundle.device)
    _log(log, "info", f"MuseTalk 推理 {n} 帧", {"frames": n})
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        z = torch.cat([cycle_z[i % len(cycle_z)] for i in range(i0, i1)], dim=0).to(device=bundle.device, dtype=bundle.dtype)
        audio = bundle.pe(chunks[i0:i1].to(bundle.device, dtype=bundle.dtype)).to(dtype=bundle.dtype)
        steps_b = steps.expand(z.shape[0])
        pred = bundle.unet(z, steps_b, encoder_hidden_states=audio).sample
        gens = _decode(pred, bundle)
        for j, gen in enumerate(gens):
            idx = (i0 + j) % len(cycle_fr)
            face = cycle_face[idx]
            out.append(_blend_mouth(cycle_fr[idx], gen, face["bbox"], face.get("pts")))
    return out


def generate_talking_photo(
    image_name: str,
    audio_name: str,
    log_callback: LogCb | None = None,
) -> dict[str, Any]:
    image_path = UPLOAD_DIR / image_name
    audio_path = UPLOAD_DIR / audio_name
    if not image_path.exists():
        raise FileNotFoundError(f"人像不存在: {image_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频不存在: {audio_path}")

    duration_s = min(MAX_DURATION_S, max(1.0, _ffprobe_duration(audio_path) or 3.0))
    _log(log_callback, "info", "MuseTalk：检测人脸", {"duration_s": duration_s})
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"读不了图片: {image_path}")
    h0, w0 = bgr.shape[:2]
    max_side = 1280
    if max(h0, w0) > max_side:
        s = max_side / float(max(h0, w0))
        w0, h0 = int(w0 * s), int(h0 * s)
    w, h = _even(max(2, w0)), _even(max(2, h0))
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    face = _detect_face(bgr)
    with tempfile.TemporaryDirectory(prefix="musetalk_") as td:
        wav = Path(td) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", "-t", str(duration_s), "-c:a", "pcm_s16le", str(wav)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        frames = _infer_frames([bgr], [face], wav, log_callback)
        encoded = Path(td) / "out.mp4"
        _save_mp4(frames, wav, encoded, FPS_OUT)
        name = _write_outputs(encoded, "talking_fast")
    _log(log_callback, "info", "MuseTalk 口型完成", {"filename": name, "duration_s": duration_s})
    return {
        "filename": name,
        "url": f"/comfy-output/{name}",
        "duration_s": duration_s,
        "backend": "musetalk15",
        "width": w,
        "height": h,
    }


def generate_talking_video(
    video_name: str,
    audio_name: str,
    log_callback: LogCb | None = None,
) -> dict[str, Any]:
    video_path = UPLOAD_DIR / video_name
    audio_path = UPLOAD_DIR / audio_name
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    duration_s = min(MAX_DURATION_S, max(1.0, _ffprobe_duration(audio_path) or 3.0))
    _log(log_callback, "info", "MuseTalk：读取视频并检测人脸", {"duration_s": duration_s})
    cap = cv2.VideoCapture(str(video_path))
    src: list[np.ndarray] = []
    try:
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            h0, w0 = fr.shape[:2]
            if max(h0, w0) > 960:
                s = 960 / float(max(h0, w0))
                fr = cv2.resize(fr, (_even(int(w0 * s)), _even(int(h0 * s))), interpolation=cv2.INTER_AREA)
            src.append(fr)
            if len(src) / float(FPS_OUT) >= MAX_DURATION_S:
                break
    finally:
        cap.release()
    if len(src) < 3:
        raise RuntimeError("视频太短或没有可读帧")
    h, w = src[0].shape[:2]
    src = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[:2] != (h, w) else f for f in src]
    last = _detect_face(src[0])
    faces = []
    for fr in src:
        try:
            last = _detect_face(fr)
        except Exception:
            pass
        faces.append(last)
    still = src[0]
    still_name = f"talkvid_still_{random.randint(100000, 999999)}.jpg"
    cv2.imwrite(str(UPLOAD_DIR / still_name), still, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    with tempfile.TemporaryDirectory(prefix="musetalkv_") as td:
        wav = Path(td) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", "-t", str(duration_s), "-c:a", "pcm_s16le", str(wav)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        frames = _infer_frames(src, faces, wav, log_callback)
        encoded = Path(td) / "out.mp4"
        _save_mp4(frames, wav, encoded, FPS_OUT)
        name = _write_outputs(encoded, "talking_video")
    _log(log_callback, "info", "MuseTalk 视频数字人完成", {"filename": name})
    return {
        "filename": name,
        "url": f"/comfy-output/{name}",
        "duration_s": duration_s,
        "backend": "musetalk15",
        "width": w,
        "height": h,
        "image_name": still_name,
        "src_frames": len(src),
    }
