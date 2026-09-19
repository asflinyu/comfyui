# coding=utf-8
"""Video generation for S2V console.

Two backends:
1. simple_s2v_api on port 7000 (AI motion; currently unstable on this env).
2. ffmpeg fallback: image + TTS audio -> MP4 with slow zoom (Ken Burns).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import utils

from config import OUTPUT_DIR, UPLOAD_DIR

SIMPLE_S2V_API = "http://127.0.0.1:7000"


def _upload_to_simple_s2v(file_path: Path, endpoint: str) -> str:
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        r = httpx.post(f"{SIMPLE_S2V_API}{endpoint}", files=files, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("filename") or data.get("image_name") or data.get("audio_name") or file_path.name


def _wait_task(tid: str, timeout: int = 600) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{SIMPLE_S2V_API}/api/status/{tid}", timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "")
        if status in ("completed", "done", "success"):
            return data
        if status in ("failed", "error"):
            raise RuntimeError(f"Video task failed: {data.get('error', data)}")
        time.sleep(2)
    raise TimeoutError(f"Video task {tid} timed out")


def _ffmpeg_fallback(image_path: Path, audio_path: Path | None, out_path: Path, duration_s: float = 3.0, width: int = 480, height: int = 832) -> None:
    """Create a video from a still image + audio using ffmpeg with a slow zoom.

    If audio is missing or its duration differs, we use the requested duration_s.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")

    # Use audio duration if available, otherwise duration_s
    target_duration = duration_s
    if audio_path and audio_path.exists():
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True, check=True,
            )
            target_duration = max(float(probe.stdout.strip()), duration_s)
        except Exception:
            pass

    audio_input = str(audio_path) if audio_path and audio_path.exists() else "anullsrc=r=44100:cl=mono"
    audio_filter = "[0:a]" if (audio_path and audio_path.exists()) else "[1:a]"

    # Pad the reference image to the target vertical ratio, then slow zoom
    total_frames = max(int(target_duration * 30), 1)
    zoom_expr = f"1+0.06*in/{total_frames}"
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
        f"zoompan=z='{zoom_expr}':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={width}x{height}:fps=30,"
        f"format=yuv420p"
    )

    cmd: list[str]
    if audio_path and audio_path.exists():
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image_path),
            "-i", str(audio_path),
            "-vf", vf,
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest", "-t", str(target_duration),
            str(out_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-loop", "1", "-i", str(image_path),
            "-vf", vf,
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest", "-t", str(target_duration),
            str(out_path),
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _concat_videos(video_paths: list[Path], out_path: Path) -> None:
    """Concatenate multiple MP4s using ffmpeg concat demuxer."""
    if not video_paths:
        raise ValueError("No videos to concatenate")
    if len(video_paths) == 1:
        shutil.copy2(video_paths[0], out_path)
        return

    list_file = out_path.with_suffix(".txt")
    list_file.write_text("\n".join(f"file '{p.absolute()}'" for p in video_paths), encoding="utf-8")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", str(out_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    finally:
        list_file.unlink(missing_ok=True)


def generate_shot_video(
    image_name: str,
    audio_name: str | None,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    duration_s: float,
    seed: int | None = None,
    mode: str = "s2v",
    use_simple_s2v: bool = False,
    log_callback: Any = None,
) -> dict[str, Any]:
    """Generate one video for a shot.

    By default uses the ffmpeg fallback (fast and reliable). Set use_simple_s2v=True
    to attempt the AI motion backend on port 7000.
    Returns {"filename": <mp4 filename in OUTPUT_DIR>, "url": ...}.
    """

    def _log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        if callable(log_callback):
            log_callback(level, msg, data)

    image_path = UPLOAD_DIR / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Shot image not found: {image_path}")
    _log("info", f"开始生成分镜视频", {"shot": image_name, "duration_s": duration_s, "mode": mode})

    suffix = Path(image_name).suffix or ".png"
    out_name = f"s2v_console_shot_{image_name.replace(suffix, '')}_{int(time.time())}.mp4"
    out_path = OUTPUT_DIR / out_name

    if use_simple_s2v:
        try:
            _log("info", "尝试 simple_s2v AI 动效后端", {"api": SIMPLE_S2V_API})
            s2v_image_name = _upload_to_simple_s2v(image_path, "/api/upload/image")
            s2v_audio_name = ""
            if audio_name:
                audio_path = UPLOAD_DIR / audio_name
                if audio_path.exists():
                    s2v_audio_name = _upload_to_simple_s2v(audio_path, "/api/upload/audio")
            payload: dict[str, Any] = {
                "image_name": s2v_image_name,
                "audio_name": s2v_audio_name or None,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "duration_s": duration_s,
                "mode": mode,
                "video_count": 1,
            }
            if seed is not None:
                payload["seed_base"] = seed
            r = httpx.post(f"{SIMPLE_S2V_API}/api/generate", json=payload, timeout=30)
            r.raise_for_status()
            tid = r.json()["task_id"]
            _log("info", "simple_s2v 任务已提交", {"task_id": tid})
            result = _wait_task(tid)
            _log("info", "simple_s2v 任务完成", {"task_id": tid, "status": result.get("status") if isinstance(result, dict) else None})
            videos = result.get("videos") if isinstance(result, dict) else None
            if videos:
                src = Path(videos[0].get("path", ""))
                if not src.exists():
                    src = OUTPUT_DIR / videos[0].get("filename", "")
                if src.exists():
                    shutil.copy2(src, out_path)
                    return {"filename": out_name, "url": f"/comfy-output/{out_name}"}
        except Exception as e:
            _log("warn", "simple_s2v 失败，改走 ComfyUI Wan", {"error": str(e)})

    try:
        from services import comfy_video

        raw = comfy_video.generate_shot_video(
            image_name=image_name,
            audio_name=audio_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            duration_s=duration_s,
            seed=seed,
            mode=mode,
            log_callback=log_callback,
        )
        _log("info", "ComfyUI 视频已生成，开始缩放/混音", {"raw": str(raw)})
        comfy_video.finalize_video(raw, out_path, width, height, audio_name, duration_s)
        _log("info", "分镜视频生成完成", {"filename": out_name, "backend": "comfy_wan"})
        return {"filename": out_name, "url": f"/comfy-output/{out_name}"}
    except Exception as e:
        _log("warn", "ComfyUI Wan 视频失败，使用 ffmpeg fallback", {"error": str(e)})

    # Fallback: ffmpeg image+audio video
    audio_path = UPLOAD_DIR / audio_name if audio_name else None
    _log("info", "使用 ffmpeg fallback 生成视频", {"image": image_name, "audio": audio_name, "out": out_name})
    _ffmpeg_fallback(image_path, audio_path, out_path, duration_s=duration_s, width=width, height=height)
    _log("info", "分镜视频生成完成", {"filename": out_name, "backend": "ffmpeg"})
    return {"filename": out_name, "url": f"/comfy-output/{out_name}"}


def generate_project_videos(
    shots: list[dict[str, Any]],
    width: int,
    height: int,
    seed: int | None = None,
    mode: str = "s2v",
    use_simple_s2v: bool = False,
    log_callback: Any = None,
) -> dict[str, Any]:
    """Generate a video per shot and optionally concatenate them.

    Returns {"videos": [...], "concat_video": {...}}.
    """

    def _log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        if callable(log_callback):
            log_callback(level, msg, data)

    # 断点：一键全程第 4 步 - 实际视频生成逻辑入口（仅在 VS Code F5 调试时生效）
    utils.debug_break()
    _log("info", "开始批量视频生成", {"shot_count": len(shots), "width": width, "height": height, "mode": mode})
    shot_videos: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    for i, shot in enumerate(shots):
        if not shot.get("image_name"):
            _log("warn", f"跳过无参考图的镜头 {shot.get('index', i + 1)}")
            continue
        _log("info", f"生成镜头 {shot.get('index', i + 1)} 视频")
        res = generate_shot_video(
            image_name=shot["image_name"],
            audio_name=shot.get("audio_name"),
            positive_prompt=shot.get("positive_prompt", ""),
            negative_prompt=shot.get("negative_prompt", ""),
            width=width,
            height=height,
            duration_s=float(shot.get("duration_s", 3.0)),
            seed=seed,
            mode=mode,
            use_simple_s2v=use_simple_s2v,
            log_callback=log_callback,
        )
        shot_videos.append({"shot_index": shot.get("index", i + 1), **res})
        video_paths.append(OUTPUT_DIR / res["filename"])

    concat_video = None
    if len(video_paths) > 1:
        concat_name = f"s2v_console_concat_{int(time.time())}.mp4"
        concat_path = OUTPUT_DIR / concat_name
        _log("info", "拼接全部镜头视频", {"count": len(video_paths), "output": concat_name})
        try:
            _concat_videos(video_paths, concat_path)
            concat_video = {"filename": concat_name, "url": f"/comfy-output/{concat_name}"}
            _log("info", "视频拼接完成", {"filename": concat_name})
        except Exception as e:
            _log("error", "视频拼接失败", {"error": str(e)})
            concat_video = {"error": str(e)}
    elif video_paths:
        _log("info", "只有一个镜头，无需拼接")

    return {"videos": shot_videos, "concat_video": concat_video}
