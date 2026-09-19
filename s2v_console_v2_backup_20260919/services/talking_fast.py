# coding=utf-8
"""蝉镜式照片口播：底图 + 音频，驱动五官（不跑 Wan 14B）。"""
from __future__ import annotations

import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from config import OUTPUT_DIR, ROOT, UPLOAD_DIR

LogCb = Callable[[str, str, dict[str, Any] | None], None]
FPS_OUT = 25
MAX_DURATION_S = 45.0

# MediaPipe FaceMesh 嘴唇 / 下巴
LIP_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
LIP_INNER = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
LIP_UPPER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78)
LIP_LOWER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78)
NOSE = 1
CHIN = 152


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


def _audio_to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _smooth(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = len(kernel) // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def _drive_signal(samples: np.ndarray, sr: int, fps: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """从音频得到开口度、嘴宽。静音闭嘴，元音张开，齿音略收。"""
    hop = max(1, int(round(sr / fps)))
    win = max(hop, int(sr * 0.045))
    rms = np.zeros(n_frames, np.float32)
    zcr = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        c = int(i * hop)
        a = max(0, c - win // 2)
        b = min(len(samples), a + win)
        chunk = samples[a:b]
        if len(chunk) < 8:
            continue
        rms[i] = float(np.sqrt(np.mean(chunk * chunk)) + 1e-8)
        zcr[i] = float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.int8)))))
    p95 = float(np.percentile(rms, 95))
    abs_floor = 0.012
    if p95 < abs_floor:
        env = np.zeros_like(rms)
    else:
        env = np.clip(rms / p95, 0.0, 1.0)
        env[rms < abs_floor] = 0.0
    z_p = float(np.percentile(zcr, 90)) or 1.0
    z = np.clip(zcr / z_p, 0.0, 1.0)
    env = _smooth(env, np.array([0.10, 0.22, 0.36, 0.22, 0.10], dtype=np.float32))
    z = _smooth(z, np.array([0.18, 0.64, 0.18], dtype=np.float32))
    env = np.clip((env - 0.03) / 0.90, 0.0, 1.0)
    attack = np.clip(np.diff(env, prepend=float(env[0])), 0.0, 1.0)
    opening = np.clip(env ** 0.50 + 0.28 * attack - 0.10 * z * env, 0.0, 1.0)
    width = np.clip(0.22 + 0.78 * opening - 0.28 * z, 0.0, 1.0)
    return opening.astype(np.float32), width.astype(np.float32)


def _envelope(samples: np.ndarray, sr: int, fps: int, n_frames: int) -> np.ndarray:
    opening, _ = _drive_signal(samples, sr, fps, n_frames)
    return opening


def _landmarks(bgr: np.ndarray) -> np.ndarray:
    import mediapipe as mp

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    try:
        res = mesh.process(rgb)
    finally:
        mesh.close()
    if not res.multi_face_landmarks:
        raise RuntimeError("照片里没有检测到人脸，请换一张清晰正脸")
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts


def _mouth_center(pts: np.ndarray) -> tuple[float, float, float, float]:
    lips = pts[list(LIP_OUTER + LIP_INNER)]
    x0, y0 = lips.min(axis=0)
    x1, y1 = lips.max(axis=0)
    cx = float((x0 + x1) * 0.5)
    cy = float((y0 + y1) * 0.5)
    mw = max(8.0, float(x1 - x0))
    mh = max(6.0, float(y1 - y0))
    return cx, cy, mw, mh


def _poly_mask(h: int, w: int, pts_xy: np.ndarray) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    poly = np.round(pts_xy).astype(np.int32)
    if len(poly) < 3:
        return mask
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _column_spans(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每列 mask 的顶/底。无效列 valid=0。"""
    h, w = mask.shape
    top = np.full(w, -1, np.int32)
    bot = np.full(w, -1, np.int32)
    valid = np.zeros(w, np.uint8)
    for x in np.flatnonzero(mask.any(axis=0)):
        ys = np.flatnonzero(mask[:, x])
        top[x] = int(ys[0])
        bot[x] = int(ys[-1])
        valid[x] = 1
    return top, bot, valid


def _edge_colors(bgr: np.ndarray, row: np.ndarray, valid: np.ndarray, delta: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    out = np.zeros((w, 3), np.float32)
    xs = np.flatnonzero(valid)
    if len(xs) == 0:
        return out
    rr = np.clip(row[xs] + delta, 0, h - 1)
    out[xs] = bgr[rr, xs]
    allx = np.arange(w)
    for c in range(3):
        out[:, c] = np.interp(allx, xs, out[xs, c])
    return out


def _prep_face(bgr: np.ndarray, pts: np.ndarray, grid: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    h, w = bgr.shape[:2]
    cx, cy, mw, mh = _mouth_center(pts)
    if grid is not None:
        xx, yy = grid
    else:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    outer_mask = _poly_mask(h, w, pts[list(LIP_OUTER)])
    inner_mask = _poly_mask(h, w, pts[list(LIP_INNER)])
    inner_mask = cv2.bitwise_and(inner_mask, outer_mask)
    if int(inner_mask.sum()) < 20:
        x0 = int(np.clip(cx - mw * 0.32, 0, w - 1))
        x1 = int(np.clip(cx + mw * 0.32, 0, w - 1))
        y0 = int(np.clip(cy, 0, h - 2))
        inner_mask[y0 : y0 + 2, x0 : x1 + 1] = 255
        inner_mask = cv2.bitwise_and(inner_mask, outer_mask)
    vermillion = cv2.subtract(outer_mask, inner_mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    upper_mask = cv2.bitwise_and(_poly_mask(h, w, pts[list(LIP_UPPER)]), vermillion)
    lower_mask = cv2.bitwise_and(_poly_mask(h, w, pts[list(LIP_LOWER)]), vermillion)
    if int(upper_mask.sum()) < 20:
        upper_mask = vermillion.copy()
        upper_mask[int(np.clip(cy, 0, h - 1)) :, :] = 0
    if int(lower_mask.sum()) < 20:
        lower_mask = vermillion.copy()
        lower_mask[: int(np.clip(cy, 0, h - 1)), :] = 0
    upper_mask = cv2.dilate(upper_mask, k3)
    lower_mask = cv2.dilate(lower_mask, k3)
    upper_mask = cv2.bitwise_and(upper_mask, cv2.dilate(outer_mask, k3))
    lower_mask = cv2.bitwise_and(lower_mask, cv2.dilate(outer_mask, k3))

    it, ib, ival = _column_spans(inner_mask)
    ot, ob, oval = _column_spans(outer_mask)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(inner_mask > 0)
    if len(ys) >= 8:
        vals = gray[ys, xs].astype(np.float32)
        mean_b, p80 = float(vals.mean()), float(np.percentile(vals, 80))
    else:
        mean_b, p80 = 40.0, 40.0
    teeth_visible = p80 >= 125.0 or mean_b >= 98.0

    inner_h = float(np.median((ib - it)[ival > 0])) if ival.any() else 2.0
    rest_open = float(np.clip(inner_h / max(mh, 4.0), 0.04, 0.90))
    max_open_px = float(max(14.0, min(0.58 * mh, 0.30 * mw, 40.0)))

    unit = float(max(mh, mw * 0.28, 10.0))
    return {
        "cx": cx,
        "cy": cy,
        "mw": mw,
        "mh": mh,
        "xx": xx,
        "yy": yy,
        "rest_open": rest_open,
        "unit": unit,
        "face_cx": float(pts[NOSE][0]),
        "face_cy": float(pts[NOSE][1]),
        "h": h,
        "w": w,
        "teeth_visible": teeth_visible,
        "inner_bright": mean_b,
        "inner_p80": p80,
        "inner_mask": inner_mask,
        "outer_mask": outer_mask,
        "vermillion": vermillion,
        "upper_a": cv2.GaussianBlur(upper_mask.astype(np.float32) / 255.0, (0, 0), 0.45),
        "lower_a": cv2.GaussianBlur(lower_mask.astype(np.float32) / 255.0, (0, 0), 0.45),
        "it": it,
        "ib": ib,
        "ot": ot,
        "ob": ob,
        "ival": ival,
        "oval": oval,
        "open_up": _edge_colors(bgr, it, ival, 0),
        "open_dn": _edge_colors(bgr, ib, ival, 0),
        "skin_up": _edge_colors(bgr, ot, oval, -2),
        "skin_dn": _edge_colors(bgr, ob, oval, 2),
        "inner_h": inner_h,
        "max_open_px": max_open_px,
        "rest_px": inner_h,
    }


def _translate(img: np.ndarray, dy: float, dx: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    if img.ndim == 2:
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    return np.clip(dst.astype(np.float32) * (1.0 - a) + src.astype(np.float32) * a, 0, 255).astype(np.uint8)


def _warp_talking(
    bgr: np.ndarray,
    face: dict[str, Any],
    amount: float,
    t: float,
    head_motion: bool = True,
    width: float | None = None,
) -> np.ndarray:
    """上下唇整片平移开合：口红纹理跟着走，但不被纵向拉开。"""
    h, w = int(face["h"]), int(face["w"])
    cx, cy = float(face["cx"]), float(face["cy"])
    mw, mh = float(face["mw"]), float(face["mh"])
    xx, yy = face["xx"], face["yy"]
    open_amt = float(np.clip(amount, 0.0, 1.0))
    wide = float(np.clip(width if width is not None else open_amt, 0.0, 1.0))
    rest_px = float(face.get("rest_px") or face.get("inner_h") or 4.0)
    max_open_px = float(face.get("max_open_px") or max(12.0, 0.42 * mh))
    target_px = 3.2 + (max_open_px - 3.2) * open_amt
    delta_px = float(np.clip(target_px - rest_px, -0.50 * mh, 0.75 * mh))
    up_dy = -0.38 * delta_px
    dn_dy = 0.72 * delta_px
    _ = wide  # 不再横向缩放，避免嘴角被扯出黑洞

    talking = bgr.copy()
    xx_i = np.clip(np.rint(xx).astype(np.int32), 0, w - 1)
    # 中间开合、嘴角不动：每列位移按到嘴角的距离衰减。
    tx = (np.arange(w, dtype=np.float32) - cx) / max(mw * 0.50, 4.0)
    fall = np.clip(1.0 - tx * tx, 0.0, 1.0)
    fall2 = fall[xx_i]
    up_col = up_dy * fall
    dn_col = dn_dy * fall

    mouth_cols = (face["ival"] > 0) | (face["oval"] > 0)
    if mouth_cols.any():
        xs = np.flatnonzero(mouth_cols)
        pad = max(3, int(round(0.06 * mw)))
        mouth_cols[max(0, int(xs[0]) - pad) : min(len(mouth_cols), int(xs[-1]) + pad + 1)] = True
    valid = mouth_cols[xx_i]
    new_it = face["it"][xx_i].astype(np.float32) + up_dy * fall2
    new_ib = face["ib"][xx_i].astype(np.float32) + dn_dy * fall2
    new_ob = face["ob"][xx_i].astype(np.float32) + dn_dy * fall2

    map_x = xx.astype(np.float32)
    map_y_u = np.clip(yy - up_col[xx_i], 0, h - 1).astype(np.float32)
    map_y_d = np.clip(yy - dn_col[xx_i], 0, h - 1).astype(np.float32)
    new_u = cv2.remap(face["upper_a"], map_x, map_y_u, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    new_d = cv2.remap(face["lower_a"], map_x, map_y_d, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    vacated_u = np.clip(face["upper_a"] - new_u, 0.0, 1.0)
    vacated_d = np.clip(face["lower_a"] - new_d, 0.0, 1.0)
    # 嘴角几乎不位移，空位用原图像素补，不要涂黑。
    corner = (fall2 < 0.22).astype(np.float32)
    vacated_u *= 1.0 - corner
    vacated_d *= 1.0 - corner
    gap = valid & (yy > new_it) & (yy < new_ib) & (fall2 > 0.20)
    on_lips = (new_u + new_d) > 0.28
    teeth = (face["inner_mask"] > 0) & (delta_px >= -1.0)
    cav_u = vacated_u * ((up_dy < -0.15) | gap).astype(np.float32)
    cav_d = vacated_d * ((dn_dy > 0.15) | gap).astype(np.float32)
    skin_u = vacated_u * (1.0 if up_dy > 0.15 else 0.0) * (~gap).astype(np.float32)
    skin_d = vacated_d * (1.0 if dn_dy < -0.15 else 0.0) * (~gap).astype(np.float32)
    cav_u[teeth] = 0
    cav_d[teeth] = 0
    extra_gap = gap.astype(np.float32) * (1.0 - on_lips.astype(np.float32))
    extra_gap[teeth] = 0
    extra_gap = cv2.GaussianBlur(extra_gap, (0, 0), 0.6)
    cav_u = np.maximum(cav_u, extra_gap * 0.85)
    cav_d = np.maximum(cav_d, extra_gap * 0.85)

    span = np.maximum(new_ib - new_it, 1.0)
    mix = np.clip((yy - new_it) / span, 0.0, 1.0)[..., None]
    cav_col = face["open_up"][xx_i] * (1.0 - mix) + face["open_dn"][xx_i] * mix
    talking = _blend(talking, cav_col, np.clip(np.maximum(cav_u, cav_d), 0, 1))
    talking = _blend(talking, face["skin_up"][xx_i], skin_u)
    talking = _blend(talking, face["skin_dn"][xx_i], skin_d)

    lip_src_u = cv2.remap(bgr, map_x, map_y_u, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lip_src_d = cv2.remap(bgr, map_x, map_y_d, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    talking = _blend(talking, lip_src_u, new_u)
    talking = _blend(talking, lip_src_d, new_d)

    if abs(dn_dy) > 0.35:
        below = (yy > new_ob + 1.5) & (np.abs(xx - cx) < mw * 1.08) & ((new_u + new_d) < 0.15)
        jaw_w = np.clip((yy - (new_ob + 1.5)) / max(mh * 2.1, 8.0), 0.0, 1.0)
        jaw_w = np.power(jaw_w, 1.15) * below.astype(np.float32)
        face_dx = np.abs(xx - float(face["face_cx"])) / max(mw * 1.6, 8.0)
        jaw_w *= np.exp(-0.6 * np.clip(face_dx, 0.0, 4.0) ** 2).astype(np.float32)
        jaw_shift = 0.55 * dn_dy * jaw_w
        if float(np.max(np.abs(jaw_shift))) > 0.2:
            frozen_a = np.clip(new_u + new_d, 0.0, 1.0)
            frozen = talking.copy()
            map_x = xx.astype(np.float32)
            map_y = np.clip(yy + jaw_shift, 0, h - 1).astype(np.float32)
            talking = cv2.remap(talking, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            talking = _blend(talking, frozen, frozen_a)

    if not head_motion:
        return talking
    angle = 0.10 * np.sin(2 * np.pi * 0.28 * t) + 0.05 * open_amt * np.sin(2 * np.pi * 1.7 * t)
    rot = cv2.getRotationMatrix2D((float(face["face_cx"]), float(face["face_cy"])), float(angle), 1.0)
    rot[0, 2] += 0.0007 * w * np.sin(2 * np.pi * 0.18 * t)
    rot[1, 2] += 0.18 * (dn_dy / max(mh, 8.0)) * mh
    return cv2.warpAffine(talking, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


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

    duration_s = _ffprobe_duration(audio_path)
    if duration_s <= 0:
        duration_s = 3.0
    duration_s = min(MAX_DURATION_S, max(1.0, duration_s))
    n_frames = max(FPS_OUT, int(round(duration_s * FPS_OUT)))

    _log(log_callback, "info", "快速口型：检测人脸并分析音频", {"duration_s": duration_s, "frames": n_frames})
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"读不了图片: {image_path}")
    h0, w0 = bgr.shape[:2]
    max_side = 1280
    if max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        w0, h0 = int(w0 * scale), int(h0 * scale)
    w, h = _even(max(2, w0)), _even(max(2, h0))
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    pts = _landmarks(bgr)
    face = _prep_face(bgr, pts)
    _log(
        log_callback,
        "info",
        "人脸已定位，开始按音频驱动口型",
        {
            "teeth_visible": bool(face["teeth_visible"]),
            "rest_open": round(float(face["rest_open"]), 3),
            "mouth": f"{face['mw']:.0f}x{face['mh']:.0f}",
        },
    )

    with tempfile.TemporaryDirectory(prefix="talkfast_") as td:
        td_path = Path(td)
        wav_path = td_path / "a.wav"
        _audio_to_wav(audio_path, wav_path)
        samples, sr = _read_pcm16(wav_path)
        env, wide = _drive_signal(samples, sr, FPS_OUT, n_frames)

        raw_yuv = td_path / "frames.raw"
        with raw_yuv.open("wb") as fh:
            for i in range(n_frames):
                t = i / float(FPS_OUT)
                frame = _warp_talking(bgr, face, float(env[i]), t, width=float(wide[i]))
                fh.write(np.ascontiguousarray(frame).tobytes())

        encoded = td_path / "raw.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS_OUT),
            "-i", str(raw_yuv),
            "-i", str(wav_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart",
            str(encoded),
        ]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
        if run.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败: {run.stderr[-400:].decode('utf-8', errors='ignore')}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (ROOT / "output").mkdir(parents=True, exist_ok=True)
        name = f"talking_fast_{random.randint(100000, 999999)}.mp4"
        dest = OUTPUT_DIR / name
        dest.write_bytes(encoded.read_bytes())
        local = ROOT / "output" / name
        if local.resolve() != dest.resolve():
            local.write_bytes(dest.read_bytes())

    _log(log_callback, "info", "快速口型完成", {"filename": name, "duration_s": duration_s, "size": f"{w}x{h}"})
    return {
        "filename": name,
        "url": f"/comfy-output/{name}",
        "duration_s": duration_s,
        "backend": "photo_drive",
        "width": w,
        "height": h,
    }


def _loop_index(i: int, n: int) -> int:
    if n <= 1:
        return 0
    cycle = 2 * (n - 1)
    k = i % cycle
    return k if k < n else cycle - k


def _resize_frame(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h0, w0 = bgr.shape[:2]
    if max(h0, w0) > max_side:
        scale = max_side / float(max(h0, w0))
        w0, h0 = int(w0 * scale), int(h0 * scale)
    w, h = _even(max(2, w0)), _even(max(2, h0))
    if (bgr.shape[1], bgr.shape[0]) != (w, h):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


def _read_video_frames(path: Path, max_s: float, max_side: int, fps_out: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"读不了视频: {path}")
    frames: list[np.ndarray] = []
    next_t = 0.0
    dt = 1.0 / float(fps_out)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if t + 1e-3 < next_t:
                continue
            frames.append(_resize_frame(frame, max_side))
            next_t += dt
            if next_t >= max_s:
                break
    finally:
        cap.release()
    if len(frames) < 3:
        raise RuntimeError("视频太短或没有可读帧，请换一段至少 1 秒的正脸视频")
    return frames


def _video_landmarks(mesh: Any, bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def _save_outputs(encoded: Path, prefix: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{random.randint(100000, 999999)}.mp4"
    dest = OUTPUT_DIR / name
    dest.write_bytes(encoded.read_bytes())
    local = ROOT / "output" / name
    if local.resolve() != dest.resolve():
        local.write_bytes(dest.read_bytes())
    return name


def generate_talking_video(
    video_name: str,
    audio_name: str,
    log_callback: LogCb | None = None,
) -> dict[str, Any]:
    """用上传视频当数字人底片，按音频驱动口型。保留原片头动，不跑 14B。"""
    import mediapipe as mp

    video_path = UPLOAD_DIR / video_name
    audio_path = UPLOAD_DIR / audio_name
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"音频不存在: {audio_path}")

    duration_s = _ffprobe_duration(audio_path)
    if duration_s <= 0:
        duration_s = 3.0
    duration_s = min(MAX_DURATION_S, max(1.0, duration_s))
    n_frames = max(FPS_OUT, int(round(duration_s * FPS_OUT)))

    _log(log_callback, "info", "快速视频数字人：抽帧并检测人脸", {"duration_s": duration_s, "frames": n_frames})
    src_frames = _read_video_frames(video_path, MAX_DURATION_S, 960, FPS_OUT)
    h, w = src_frames[0].shape[:2]
    for i, fr in enumerate(src_frames):
        if fr.shape[:2] != (h, w):
            src_frames[i] = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)

    grid = tuple(np.mgrid[0:h, 0:w].astype(np.float32)[::-1])
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    last_pts: np.ndarray | None = None
    teeth_lock: bool | None = None
    poster = src_frames[0]
    try:
        for fr in src_frames[: min(40, len(src_frames))]:
            pts = _video_landmarks(mesh, fr)
            if pts is not None:
                last_pts = pts
                poster = fr
                break
        if last_pts is None:
            raise RuntimeError("视频里没有检测到人脸，请换一段清晰正脸、脸不要太小")
        mesh.close()
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        still_name = f"talkvid_still_{random.randint(100000, 999999)}.jpg"
        cv2.imwrite(str(UPLOAD_DIR / still_name), poster, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        with tempfile.TemporaryDirectory(prefix="talkvid_") as td:
            td_path = Path(td)
            wav_path = td_path / "a.wav"
            _audio_to_wav(audio_path, wav_path)
            samples, sr = _read_pcm16(wav_path)
            env, wide = _drive_signal(samples, sr, FPS_OUT, n_frames)

            raw_yuv = td_path / "frames.raw"
            n_src = len(src_frames)
            rest_lock: float | None = None
            with raw_yuv.open("wb") as fh:
                for i in range(n_frames):
                    t = i / float(FPS_OUT)
                    src = src_frames[_loop_index(i, n_src)]
                    pts = _video_landmarks(mesh, src)
                    if pts is not None:
                        last_pts = pts
                    face = _prep_face(src, last_pts, grid=grid)
                    if teeth_lock is None:
                        teeth_lock = bool(face["teeth_visible"])
                        rest_lock = float(face["rest_px"])
                    else:
                        face["teeth_visible"] = teeth_lock
                        if rest_lock is not None:
                            face["rest_px"] = rest_lock
                            face["rest_open"] = rest_lock / max(float(face["mh"]), 4.0)
                    frame = _warp_talking(
                        src, face, float(env[i]), t, head_motion=False, width=float(wide[i])
                    )
                    fh.write(np.ascontiguousarray(frame).tobytes())

            encoded = td_path / "raw.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS_OUT),
                "-i", str(raw_yuv),
                "-i", str(wav_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-shortest",
                "-movflags", "+faststart",
                str(encoded),
            ]
            run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
            if run.returncode != 0:
                raise RuntimeError(f"ffmpeg 编码失败: {run.stderr[-400:].decode('utf-8', errors='ignore')}")
            name = _save_outputs(encoded, "talking_video")
    finally:
        mesh.close()

    _log(log_callback, "info", "视频数字人完成", {"filename": name, "duration_s": duration_s, "size": f"{w}x{h}"})
    return {
        "filename": name,
        "url": f"/comfy-output/{name}",
        "duration_s": duration_s,
        "backend": "video_drive",
        "width": w,
        "height": h,
        "image_name": still_name,
        "src_frames": len(src_frames),
    }
