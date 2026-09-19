# coding=utf-8
"""S2V 分镜控制台 V2 — 自包含实现。"""
from __future__ import annotations

import csv
import io
import json
import shutil
import uuid

import utils
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import openpyxl
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from config import (
    COMFY_ROOT,
    COMFY_URL,
    CONSOLE_HOST,
    CONSOLE_PORT,
    DATA_DIR,
    DEFAULT_NEGATIVE,
    DEFAULT_VOICE,
    FPS,
    FLUX_HEIGHT,
    FLUX_WIDTH,
    LLM_API_KEY,
    I2V_CLIP,
    I2V_LORA,
    I2V_UNET,
    I2V_VAE,
    INPUT_DIR,
    MAX_SHOTS,
    MAX_VIDEOS,
    MIN_SHOTS,
    OUTPUT_DIR,
    QUALITY_PRESETS,
    RESOLUTION_PRESETS,
    ROOT,
    S2V_AUDIO_ENC,
    S2V_CLIP,
    S2V_LORA,
    S2V_UNET,
    S2V_VAE,
    SD_CKPT,
    TTS_URL,
    UPLOAD_DIR,
)
from services import comfy_image, comfy_video, deepseek, talking_fast, tts, video_gen
from services.comfy_runtime import GPU

app = FastAPI(title="S2V 分镜控制台 V2", version="2.0.1")
STATIC = Path(__file__).resolve().parent / "static"


# Custom static handler for index.html with no-cache headers to prevent stale JS
@app.get("/static/index.html", response_class=HTMLResponse)
def api_index_html() -> HTMLResponse:
    content = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/t2v.html", response_class=HTMLResponse)
def api_t2v_html() -> HTMLResponse:
    content = (STATIC / "t2v.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/f2v.html", response_class=HTMLResponse)
def api_f2v_html() -> HTMLResponse:
    content = (STATIC / "f2v.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/talking.html", response_class=HTMLResponse)
def api_talking_html() -> HTMLResponse:
    content = (STATIC / "talking.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/talking_video.html", response_class=HTMLResponse)
def api_talking_video_html() -> HTMLResponse:
    content = (STATIC / "talking_video.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/s2v.html", response_class=HTMLResponse)
def api_s2v_html() -> HTMLResponse:
    content = (STATIC / "s2v.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/home.html", response_class=HTMLResponse)
def api_home_html() -> HTMLResponse:
    content = (STATIC / "home.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/t2i.html", response_class=HTMLResponse)
def api_t2i_html() -> HTMLResponse:
    content = (STATIC / "t2i.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/i2v.html", response_class=HTMLResponse)
def api_i2v_html() -> HTMLResponse:
    content = (STATIC / "i2v.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/static/layout.css")
def api_layout_css() -> FileResponse:
    return FileResponse(
        STATIC / "layout.css",
        media_type="text/css",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


def _html_redirect(path: str, request: Request) -> RedirectResponse:
    q = request.url.query
    return RedirectResponse(url=f"{path}?{q}" if q else path)


@app.get("/")
def root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/home.html", request)


@app.get("/home")
def home_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/home.html", request)


@app.get("/console")
def console_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/index.html", request)


@app.get("/img")
def img_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/t2i.html", request)


@app.get("/t2i")
def t2i_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/t2i.html", request)


@app.get("/i2v")
def i2v_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/i2v.html", request)


@app.get("/t2v")
def t2v_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/t2v.html", request)


@app.get("/f2v")
def f2v_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/f2v.html", request)


@app.get("/talking")
def talking_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/talking.html", request)


@app.get("/talking-video")
def talking_video_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/talking_video.html", request)


@app.get("/s2v")
def s2v_root(request: Request) -> RedirectResponse:
    return _html_redirect("/static/s2v.html", request)


# In-memory task store for generation status
tasks: dict[str, dict[str, Any]] = {}


def _random_id() -> str:
    return uuid.uuid4().hex[:10]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_log(tid: str, level: str, message: str, data: dict[str, Any] | None = None) -> None:
    """Append a structured log entry to a background task if it exists."""
    task = tasks.get(tid)
    if not task:
        return
    if "logs" not in task:
        task["logs"] = []
    entry: dict[str, Any] = {"ts": _now(), "level": level, "message": message}
    if data:
        entry["data"] = data
    task["logs"].append(entry)


def _comfy_status() -> str:
    try:
        r = httpx.get(f"{COMFY_URL}/system_stats", timeout=5)
        return "ok" if r.status_code == 200 else "error"
    except Exception:
        return "offline"


def _tts_status() -> str:
    try:
        r = httpx.get(f"{TTS_URL}/", timeout=5)
        return "ok" if r.status_code in (200, 404) else "error"
    except Exception:
        return "offline"


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {
        "ok": True,
        "comfy_ok": _comfy_status() == "ok",
        "comfy": _comfy_status(),
        "tts": _tts_status(),
        "llm_configured": bool(LLM_API_KEY),
        "llm_provider": "moonshot" if config.MOONSHOT_API_KEY else "deepseek",
        "min_shots": MIN_SHOTS,
        "max_shots": MAX_SHOTS,
        "max_videos": MAX_VIDEOS,
        "fps": FPS,
        "presets": RESOLUTION_PRESETS,
        "quality_presets": QUALITY_PRESETS,
        "gpu": {**comfy_video.gpu_status(), **GPU.snapshot()},
        "t2v": True,
        "f2v": True,
        "s2v": True,
        "talking": True,
        "talking_video": True,
        "t2i": True,
        "i2v": True,
        "image_workflows": True,
    }


@app.get("/api/voices")
def api_voices() -> dict[str, Any]:
    """Return all Qwen3-TTS VoiceDesign presets (50). Same model, no extra VRAM per voice."""
    import importlib.util

    voices: list[str] = []
    try:
        r = httpx.get(f"{TTS_URL}/voices", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                voices = [str(x) for x in data if str(x).strip()]
            elif isinstance(data, dict) and "voices" in data:
                voices = [str(x) for x in (data.get("voices") or []) if str(x).strip()]
    except Exception:
        pass
    if len(voices) <= 1:
        try:
            path = COMFY_ROOT / "ttsvoice" / "voices.py"
            spec = importlib.util.spec_from_file_location("s2v_tts_voices", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                names = list(mod.voice_choices() or [])
                if names:
                    voices = names
        except Exception:
            pass
    if not voices:
        voices = [DEFAULT_VOICE]
    if DEFAULT_VOICE in voices:
        voices = [DEFAULT_VOICE] + [v for v in voices if v != DEFAULT_VOICE]
    return {"voices": voices, "count": len(voices)}


class OptimizeReq(BaseModel):
    brief: str = ""
    target_seconds: int = Field(15, ge=6, le=60)
    preferred_shots: int = Field(8, ge=MIN_SHOTS, le=MAX_SHOTS)
    mode: str = "i2v"


@app.post("/api/optimize")
def api_optimize(req: OptimizeReq) -> dict[str, Any]:
    if not LLM_API_KEY:
        raise HTTPException(503, "未配置 LLM API key")
    try:
        optimized = deepseek.optimize_brief(
            req.brief,
            mode=req.mode,
            target_seconds=req.target_seconds,
            preferred_shots=req.preferred_shots,
        )
        return {"optimized": optimized}
    except Exception as e:
        raise HTTPException(500, f"DeepSeek optimize failed: {e}")


class StoryboardReq(BaseModel):
    brief: str = ""
    target_seconds: int = Field(15, ge=6, le=60)
    preferred_shots: int = Field(8, ge=MIN_SHOTS, le=MAX_SHOTS)
    mode: str = "i2v"
    project: dict[str, Any] | None = None
    revise: bool = False


@app.post("/api/storyboard")
def api_storyboard(req: StoryboardReq) -> dict[str, Any]:
    # 断点：一键全程第 1 步 - AI 分镜入口（仅在 VS Code F5 调试时生效）
    utils.debug_break()
    if not LLM_API_KEY:
        raise HTTPException(503, "未配置 LLM API key")
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.info(f"[storyboard] brief_len={len(req.brief)} mode={req.mode} shots={req.preferred_shots}")
        project = deepseek.generate_storyboard(
            req.brief,
            mode=req.mode,
            target_seconds=req.target_seconds,
            preferred_shots=req.preferred_shots,
            project=req.project,
            revise=req.revise,
        )
        logging.info(f"[storyboard] result title={project.get('title')} shots={len(project.get('shots', []))}")
        _save_project(project)
        return project
    except Exception as e:
        import logging, traceback
        logging.error(f"[storyboard] error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"AI 分镜生成失败: {e}")


class SaveReq(BaseModel):
    project: dict[str, Any]


@app.post("/api/project/save")
def api_project_save(req: SaveReq) -> dict[str, Any]:
    project = req.project
    if not project.get("id"):
        project["id"] = _random_id()
    project["updated_at"] = _now()
    _save_project(project)
    return project


def _save_project(project: dict[str, Any]) -> None:
    pid = project.get("id")
    if not pid:
        return
    path = DATA_DIR / f"{pid}.json"
    path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


@app.post("/api/upload/image")
def api_upload_image(
    files: list[UploadFile] = File(default_factory=list),
    shot_index: int = Form(0),
    project_id: str = Form(""),
) -> dict[str, Any]:
    results = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix or ".png"
        name = f"console_{project_id}_shot{shot_index}_{_random_id()}{suffix}"
        dest = UPLOAD_DIR / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        results.append({"shot_index": shot_index, "image_name": name})
    return {"results": results, "error": "" if results else "No file uploaded"}


def _probe_audio_duration(path: Path) -> float:
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            text=True,
            timeout=10,
        )
        return max(0.0, float((out or "0").strip() or 0))
    except Exception:
        return float(tts._audio_duration(path) or 0.0)


@app.post("/api/upload/audio")
def api_upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    orig = file.filename or "audio.wav"
    suffix = Path(orig).suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}:
        raise HTTPException(400, "音频仅支持 wav / mp3 / flac / m4a / ogg")
    name = f"s2v_audio_{_random_id()}{suffix}"
    dest = UPLOAD_DIR / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    duration_s = round(_probe_audio_duration(dest), 2)
    return {"audio_name": name, "duration_s": duration_s, "error": ""}


@app.post("/api/upload/video")
def api_upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    orig = file.filename or "clip.mp4"
    suffix = Path(orig).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}:
        raise HTTPException(400, "视频仅支持 mp4 / webm / mov / mkv / avi")
    name = f"talkvid_{_random_id()}{suffix}"
    dest = UPLOAD_DIR / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    duration_s = round(_probe_audio_duration(dest), 2)
    return {"video_name": name, "duration_s": duration_s, "error": ""}


def _extract_audio_from_video(video_path: Path) -> tuple[str, float]:
    import subprocess

    name = f"s2v_audio_{_random_id()}.wav"
    dest = UPLOAD_DIR / name
    run = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dest),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if run.returncode != 0 or not dest.exists() or dest.stat().st_size < 2000:
        if dest.exists():
            dest.unlink()
        raise ValueError("视频没有可用音轨。请填写台词，或另传一段音频")
    duration_s = _probe_audio_duration(dest)
    if duration_s < 0.4:
        dest.unlink(missing_ok=True)
        raise ValueError("视频音轨太短。请填写台词，或另传一段音频")
    return name, duration_s


def _parse_excel(file_bytes: bytes, filename: str) -> str:
    """Parse an Excel/CSV file into a readable script text."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        reader = csv.reader(io.StringIO(file_bytes.decode("utf-8-sig", errors="ignore")))
        rows = list(reader)
    else:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        rows: list[list[Any]] = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                rows.append(list(row))

    lines: list[str] = []
    current_title: str = ""
    current_lines: list[str] = []
    dumped: list[str] = []

    def flush() -> None:
        if current_title or current_lines:
            if current_title:
                lines.append(current_title)
            lines.extend(current_lines)
            lines.append("")

    header_words = {"#", "序号", "镜头", "时间", "画面", "台词", "反转"}
    for row in rows:
        raw = ["" if c is None else str(c).strip() for c in row]
        cells = [c for c in raw if c]
        if not cells:
            continue
        dumped.append(" | ".join(cells))
        first = cells[0]
        joined = "".join(cells)
        if any(k in joined for k in ("8镜头8反转",)):
            continue
        if all(c in header_words or c.replace(" ", "") in header_words for c in cells):
            continue
        if first in header_words and len(cells) <= 5 and any(c in header_words for c in cells[1:]):
            continue
        if len(cells) == 1:
            flush()
            current_title = first
            current_lines = []
            continue
        if len(cells) >= 4:
            idx, time_col, visual, dialogue = cells[0], cells[1], cells[2], cells[3]
            twist = cells[4] if len(cells) > 4 else ""
            line = f"镜头{idx}（{time_col}）：画面：{visual}。台词：{dialogue}"
            if twist:
                line += f"。反转：{twist}"
            current_lines.append(line)
        else:
            current_lines.append("；".join(cells))
    flush()
    text = "\n".join(lines).strip()
    if not text:
        text = "\n".join(dumped).strip()
    return text


@app.post("/api/upload/scripts")
def api_upload_scripts(
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Upload multiple script Excel/CSV files and return parsed text per file."""
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in (".xlsx", ".xls", ".csv"):
            errors.append(f"{f.filename}: 仅支持 .xlsx/.xls/.csv")
            continue
        try:
            data = f.file.read()
            text = _parse_excel(data, f.filename)
            if text:
                results.append({"filename": f.filename, "text": text})
            else:
                errors.append(f"{f.filename}: 未解析到有效文案")
        except Exception as e:
            errors.append(f"{f.filename}: 解析失败 {e}")
    if not results and errors:
        return {"results": results, "error": "；".join(errors)}
    return {"results": results, "error": "；".join(errors) if errors else ""}


@app.post("/api/image/generate")
def api_image_generate(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a single reference image via ComfyUI T2I."""
    prompt = payload.get("prompt") or payload.get("image_prompt") or ""
    prompt_en = payload.get("prompt_en") or payload.get("image_prompt_en") or ""
    negative = payload.get("negative") or payload.get("negative_prompt") or ""
    width = int(payload.get("width", 256))
    height = int(payload.get("height", 256))
    seed = payload.get("seed")
    filename_prefix = payload.get("filename_prefix", "console_ref")
    source_image_name = payload.get("source_image_name") or payload.get("image_name") or ""
    denoise = float(payload.get("denoise", 0.65))
    if not prompt and not prompt_en:
        raise HTTPException(400, "prompt/image_prompt or prompt_en/image_prompt_en is required")
    try:
        image_name = comfy_image.generate_image(
            prompt,
            prompt_en=prompt_en,
            negative=negative,
            width=width,
            height=height,
            seed=seed,
            filename_prefix=filename_prefix,
            source_image_name=source_image_name,
            denoise=denoise,
        )
        return {"image_name": image_name, "error": ""}
    except Exception as e:
        raise HTTPException(500, f"Image generation failed: {e}")


class TTSReq(BaseModel):
    dialogue: str
    voice: str = DEFAULT_VOICE
    target_duration_s: float = 3.0
    shot_index: int = 0
    project_id: str = ""


class GenerateAllImagesReq(BaseModel):
    project: dict[str, Any]
    width: int = 1080
    height: int = 1920


def _run_images_generation(tid: str, req: GenerateAllImagesReq) -> None:
    """Background thread: batch generate reference images with progress."""
    # 断点：一键全程第 2 步 - 批量生图后台线程入口（仅在 VS Code F5 调试时生效）
    utils.debug_break()
    project = req.project
    shots = project.get("shots", [])
    pid = project.get("id", "")
    total = len(shots)
    errors: list[str] = []
    tasks[tid] = {"status": "running", "progress": 0, "message": f"准备生成 {total} 张参考图...", "results": [], "logs": []}

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    with GPU.hold(log, kind="image"):
        for i, shot in enumerate(shots):
            idx = shot.get("index", i + 1)
            if shot.get("image_name"):
                tasks[tid]["progress"] = int(((i + 1) / max(total, 1)) * 100)
                tasks[tid]["message"] = f"镜头 {idx}/{total} 已有参考图，跳过"
                continue
            prompt = shot.get("image_prompt") or ""
            prompt_en = shot.get("image_prompt_en") or shot.get("positive_prompt", "")
            negative = shot.get("negative_prompt", "")
            if not prompt and not prompt_en:
                errors.append(f"镜头 {idx} 没有参考图描述")
                continue
            tasks[tid]["progress"] = int((i / max(total, 1)) * 100)
            tasks[tid]["message"] = f"正在生成镜头 {idx}/{total} 参考图..."
            try:
                image_name = comfy_image.generate_image(
                    prompt,
                    prompt_en=prompt_en,
                    negative=negative,
                    width=req.width,
                    height=req.height,
                    filename_prefix=f"console_{pid}_shot{idx}",
                    log_callback=log,
                )
                shot["image_name"] = image_name
                shot["image_source"] = "gen"
                tasks[tid]["progress"] = int(((i + 1) / max(total, 1)) * 100)
                tasks[tid]["message"] = f"镜头 {idx}/{total} 参考图完成"
            except Exception as e:
                _task_log(tid, "error", f"镜头 {idx} 参考图生成失败: {e}")
                errors.append(f"镜头 {idx}: {e}")
    project["updated_at"] = _now()
    _save_project(project)
    tasks[tid].update({
        "status": "done",
        "progress": 100,
        "message": f"参考图完成 {total - len(errors)}/{total}" + (f"，{len(errors)} 个失败" if errors else ""),
        "project": project,
        "errors": errors,
    })


@app.post("/api/images/generate_all")
def api_images_generate_all(req: GenerateAllImagesReq) -> dict[str, Any]:
    """Batch generate reference images (async task with progress polling)."""
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    t = threading.Thread(target=_run_images_generation, args=(tid, req), daemon=True)
    t.start()
    return {"task_id": tid, "status": "queued"}


@app.post("/api/tts/shot")
def api_tts_shot(req: TTSReq) -> dict[str, Any]:
    if not req.dialogue.strip():
        raise HTTPException(400, "Dialogue is empty")
    try:
        audio_name, duration_s = tts.synthesize(req.dialogue, voice=req.voice)
        return {"audio_name": audio_name, "duration_s": duration_s, "error": ""}
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")


class TTSAllReq(BaseModel):
    project: dict[str, Any]
    voice: str = DEFAULT_VOICE
    skip_existing: bool = True


def _run_tts_generation(tid: str, req: TTSAllReq) -> None:
    """Background thread: batch synthesize TTS audio with progress."""
    # 断点：一键全程第 3 步 - 批量配音后台线程入口（仅在 VS Code F5 调试时生效）
    utils.debug_break()
    project = req.project
    shots = project.get("shots", [])
    total = len(shots)
    errors: list[str] = []
    tasks[tid] = {"status": "running", "progress": 0, "message": f"准备合成 {total} 条配音...", "results": [], "logs": []}
    for i, shot in enumerate(shots):
        idx = shot.get("index", i + 1)
        if req.skip_existing and shot.get("audio_name"):
            tasks[tid]["progress"] = int(((i + 1) / max(total, 1)) * 100)
            tasks[tid]["message"] = f"镜头 {idx}/{total} 已有音频，跳过"
            continue
        dialogue = shot.get("dialogue", "")
        if not dialogue.strip():
            tasks[tid]["progress"] = int(((i + 1) / max(total, 1)) * 100)
            tasks[tid]["message"] = f"镜头 {idx}/{total} 无台词，跳过"
            continue
        tasks[tid]["progress"] = int((i / max(total, 1)) * 100)
        tasks[tid]["message"] = f"正在合成镜头 {idx}/{total} 配音..."
        try:
            audio_name, duration_s = tts.synthesize(dialogue, voice=req.voice)
            shot["audio_name"] = audio_name
            shot["audio_duration_s"] = duration_s
            tasks[tid]["progress"] = int(((i + 1) / max(total, 1)) * 100)
            tasks[tid]["message"] = f"镜头 {idx}/{total} 配音完成"
        except Exception as e:
            errors.append(f"镜头 {idx}: {e}")
    project["updated_at"] = _now()
    _save_project(project)
    tasks[tid].update({
        "status": "done",
        "progress": 100,
        "message": f"配音完成 {total - len(errors)}/{total}" + (f"，{len(errors)} 个失败" if errors else ""),
        "project": project,
        "errors": errors,
    })


@app.post("/api/tts/all")
def api_tts_all(req: TTSAllReq) -> dict[str, Any]:
    """Batch synthesize TTS audio (async task with progress polling)."""
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    t = threading.Thread(target=_run_tts_generation, args=(tid, req), daemon=True)
    t.start()
    return {"task_id": tid, "status": "queued"}


class GenerateReq(BaseModel):
    project: dict[str, Any]
    resolution: str = "draft_480"
    video_count: int = 1
    seed_base: int | None = None
    voice: str = DEFAULT_VOICE
    skip_tts_if_audio: bool = True
    mode: str = "i2v"
    quality: str = "fast"


def _run_video_generation(tid: str, req: GenerateReq) -> None:
    """Background thread: generate a video per shot (fallback ffmpeg) and concatenate."""
    # 断点：一键全程第 4 步 - 批量视频后台线程入口（仅在 VS Code F5 调试时生效）
    utils.debug_break()
    tasks[tid] = {"status": "running", "progress": 0, "message": "开始生成视频", "results": [], "logs": []}
    project = req.project
    shots = project.get("shots", [])
    valid_shots = [s for s in shots if s.get("image_name")]
    preset = req.resolution
    width, height = 480, 832
    if preset and preset in RESOLUTION_PRESETS:
        width = RESOLUTION_PRESETS[preset]["width"]
        height = RESOLUTION_PRESETS[preset]["height"]

    total = len(valid_shots)
    tasks[tid]["message"] = f"准备生成 {total} 个镜头视频..."
    try:
        result = video_gen.generate_project_videos(
            shots=valid_shots,
            width=width,
            height=height,
            seed=req.seed_base,
            mode=req.mode,
            quality=req.quality if req.quality in QUALITY_PRESETS else "fast",
            use_simple_s2v=False,
            log_callback=lambda level, msg, data=None: _task_log(tid, level, msg, data),
        )
        videos = []
        if result.get("concat_video"):
            videos.append(result["concat_video"])
        videos.extend(result.get("videos", []))
        tasks[tid] = {
            "status": "done",
            "progress": 100,
            "message": f"完成 {len(result['videos'])}/{total} 个镜头视频",
            "results": [{"videos": videos}],
            "logs": [{"ts": _now(), "level": "info", "message": "视频生成完成"}],
        }
    except Exception as e:
        tasks[tid] = {
            "status": "error",
            "progress": 0,
            "message": f"视频生成失败: {e}",
            "results": [],
            "logs": [{"ts": _now(), "level": "error", "message": str(e)}],
        }


@app.post("/api/generate/videos")
def api_generate_videos(req: GenerateReq) -> dict[str, Any]:
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    t = threading.Thread(target=_run_video_generation, args=(tid, req), daemon=True)
    t.start()
    return {"task_id": tid, "status": "queued"}


T2V_SIZES = {
    "vertical_1080": (1080, 1920),
    "final_720": (720, 1280),
    "landscape_1080": (1920, 1080),
}


class T2VReq(BaseModel):
    prompt: str
    negative: str = ""
    resolution: str = "vertical_1080"
    duration_s: float = Field(3.0, ge=1.0, le=5.0)
    quality: str = "fast"
    seed: int | None = None
    dialogue: str = ""


def _t2v_make_audio(dialogue: str, duration_s: float) -> str | None:
    text = (dialogue or "").strip()
    if not text:
        return None
    import asyncio
    import subprocess

    import edge_tts

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    mp3 = UPLOAD_DIR / f"t2v_voice_{_random_id()}.mp3"
    wav = UPLOAD_DIR / f"t2v_voice_{_random_id()}.wav"

    async def _speak() -> None:
        comm = edge_tts.Communicate(text, "zh-CN-XiaoyiNeural", rate="+10%", pitch="+8Hz")
        await comm.save(str(mp3))

    asyncio.run(_speak())
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(mp3),
            "-af", f"apad=pad_dur={duration_s + 0.3}",
            "-t", f"{duration_s}",
            "-ar", "44100", "-ac", "1", str(wav),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return wav.name


def _run_t2v(tid: str, req: T2VReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "提交纯文字 T2V", "results": [], "logs": []}
    width, height = T2V_SIZES.get(req.resolution, (1080, 1920))
    quality = req.quality if req.quality in QUALITY_PRESETS else "fast"
    try:
        audio_name = None
        if (req.dialogue or "").strip():
            _task_log(tid, "info", "开始合成配音")
            audio_name = _t2v_make_audio(req.dialogue, req.duration_s)
            _task_log(tid, "info", "配音就绪", {"audio": audio_name})
        path = comfy_video.generate_t2v_video(
            positive_prompt=req.prompt,
            negative_prompt=req.negative,
            width=width,
            height=height,
            duration_s=req.duration_s,
            seed=req.seed,
            quality=quality,
            audio_name=audio_name,
            log_callback=lambda level, msg, data=None: _task_log(tid, level, msg, data),
        )
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {path.name}"
        tasks[tid]["results"] = [{"videos": [path.name]}]
        _task_log(tid, "info", "T2V 完成", {"filename": path.name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"T2V 失败: {e}"
        _task_log(tid, "error", str(e))


@app.post("/api/t2v/generate")
def api_t2v_generate(req: T2VReq) -> dict[str, Any]:
    if not (req.prompt or "").strip():
        raise HTTPException(400, "提示词不能为空")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_t2v, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


class F2VReq(BaseModel):
    image_prompt: str
    video_prompt: str = ""
    negative: str = ""
    duration_s: float = Field(2.0, ge=1.0, le=4.0)
    seed: int | None = None


def _run_f2v(tid: str, req: F2VReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "Flux 出底图 1280×2304", "results": [], "logs": []}
    prompt = (req.image_prompt or "").strip()
    motion = (req.video_prompt or "").strip() or prompt
    negative = req.negative or ""

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        log("info", "Flux 开始出 1280×2304 底图")
        image_name = comfy_image.generate_image(
            prompt,
            prompt_en=prompt,
            negative=negative,
            width=FLUX_WIDTH,
            height=FLUX_HEIGHT,
            seed=req.seed,
            filename_prefix="f2v_flux",
            log_callback=log,
        )
        tasks[tid]["progress"] = 40
        log("info", "Flux 底图完成，开始 fp8 4 步 I2V", {"image": image_name})
        res = video_gen.generate_shot_video(
            image_name=image_name,
            audio_name=None,
            positive_prompt=motion,
            negative_prompt=negative,
            width=1088,
            height=1920,
            duration_s=req.duration_s,
            seed=req.seed,
            mode="i2v",
            quality="fast",
            log_callback=log,
        )
        video_name = res["filename"]
        local_dir = ROOT / "output"
        local_dir.mkdir(parents=True, exist_ok=True)
        src = OUTPUT_DIR / video_name
        if src.exists():
            shutil.copy2(src, local_dir / video_name)
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {video_name}"
        tasks[tid]["results"] = [{"image": image_name, "video": video_name}]
        log("info", "F2V 完成", {"image": image_name, "video": video_name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"F2V 失败: {e}"
        _task_log(tid, "error", str(e))


@app.post("/api/f2v/generate")
def api_f2v_generate(req: F2VReq) -> dict[str, Any]:
    if not (req.image_prompt or "").strip():
        raise HTTPException(400, "出图提示词不能为空")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_f2v, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


class T2IReq(BaseModel):
    prompt: str = ""
    negative: str = ""
    seed: int | None = None
    workflow: str = "flux_dev"
    image_name: str = ""
    denoise: float = Field(0.65, ge=0.05, le=1.0)
    width: int | None = None
    height: int | None = None
    count: int = Field(1, ge=1, le=4)


def _run_t2i(tid: str, req: T2IReq) -> None:
    wf = (req.workflow or "flux_dev").strip()
    catalog = {item["id"]: item for item in comfy_image.list_workflows()}
    meta = catalog.get(wf) or catalog["flux_dev"]
    tasks[tid] = {"status": "running", "progress": 5, "message": f"生图：{meta['label']}", "results": [], "logs": []}
    prompt = (req.prompt or "").strip()
    negative = req.negative or ""
    image_name = (req.image_name or "").strip()

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        if meta.get("needs_image") and not image_name:
            raise ValueError(f"{meta['label']} 需要先上传底图")
        if wf != "upscale" and not prompt:
            raise ValueError("请填写提示词")
        count = 1 if wf == "upscale" else max(1, min(4, int(req.count or 1)))
        log("info", f"开始 {meta['label']}", {"workflow": wf, "image": image_name or None, "count": count})
        names = comfy_image.generate_images(
            prompt,
            prompt_en=prompt,
            negative=negative,
            width=int(req.width or meta["width"] or 1024),
            height=int(req.height or meta["height"] or 1024),
            seed=req.seed,
            filename_prefix=f"t2i_{wf}",
            source_image_name=image_name,
            denoise=req.denoise,
            workflow=wf,
            log_callback=log,
            count=count,
        )
        out_name = names[0]
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {len(names)} 张"
        tasks[tid]["results"] = [{"image": out_name, "images": names, "workflow": wf}]
        log("info", "生图完成", {"images": names, "workflow": wf})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"生图失败: {e}"
        _task_log(tid, "error", str(e))


@app.get("/api/image/workflows")
def api_image_workflows() -> dict[str, Any]:
    return {"workflows": comfy_image.list_workflows()}


@app.post("/api/t2i/generate")
def api_t2i_generate(req: T2IReq) -> dict[str, Any]:
    wf = (req.workflow or "flux_dev").strip()
    catalog = {item["id"]: item for item in comfy_image.list_workflows()}
    if wf not in catalog:
        raise HTTPException(400, f"未知生图工作流: {wf}")
    meta = catalog[wf]
    if meta.get("needs_image") and not (req.image_name or "").strip():
        raise HTTPException(400, f"{meta['label']} 需要先上传底图")
    if wf != "upscale" and not (req.prompt or "").strip():
        raise HTTPException(400, "出图提示词不能为空")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_t2i, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


class I2VReq(BaseModel):
    image_name: str
    video_prompt: str = ""
    negative: str = ""
    duration_s: float = Field(2.0, ge=1.0, le=4.0)
    seed: int | None = None


def _run_i2v(tid: str, req: I2VReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "提交图生视频 I2V", "results": [], "logs": []}
    image_name = (req.image_name or "").strip()
    motion = (req.video_prompt or "").strip()
    negative = req.negative or ""

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        if not image_name:
            raise ValueError("请先上传或选择底图")
        if not (UPLOAD_DIR / image_name).exists():
            raise FileNotFoundError(f"底图不存在: {image_name}")
        if not motion:
            raise ValueError("请填写动作提示词")
        log("info", "开始 fp8 4 步 I2V", {"image": image_name})
        res = video_gen.generate_shot_video(
            image_name=image_name,
            audio_name=None,
            positive_prompt=motion,
            negative_prompt=negative,
            width=1088,
            height=1920,
            duration_s=req.duration_s,
            seed=req.seed,
            mode="i2v",
            quality="fast",
            log_callback=log,
        )
        video_name = res["filename"]
        local_dir = ROOT / "output"
        local_dir.mkdir(parents=True, exist_ok=True)
        src = OUTPUT_DIR / video_name
        if src.exists():
            shutil.copy2(src, local_dir / video_name)
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {video_name}"
        tasks[tid]["results"] = [{"image": image_name, "video": video_name}]
        log("info", "I2V 完成", {"image": image_name, "video": video_name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"I2V 失败: {e}"
        _task_log(tid, "error", str(e))


@app.post("/api/i2v/generate")
def api_i2v_generate(req: I2VReq) -> dict[str, Any]:
    if not (req.image_name or "").strip():
        raise HTTPException(400, "请先上传底图")
    if not (req.video_prompt or "").strip():
        raise HTTPException(400, "动作提示词不能为空")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_i2v, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


class S2VReq(BaseModel):
    image_name: str = ""
    image_prompt: str = ""
    audio_name: str = ""
    dialogue: str = ""
    voice: str = DEFAULT_VOICE
    video_prompt: str = ""
    negative: str = ""
    seed: int | None = None


def _run_s2v(tid: str, req: S2VReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "准备口播数字人", "results": [], "logs": []}
    default_motion = (
        "close-up talking head facing the camera, the person is actively speaking, "
        "clear lip sync to the speech, mouth opening and closing with each syllable, "
        "jaw moving, natural facial expression while talking, slight head and shoulder motion, "
        "photorealistic, sharp face"
    )
    motion = (req.video_prompt or "").strip() or default_motion
    negative = req.negative or ""

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        image_name = (req.image_name or "").strip()
        if not image_name:
            prompt = (req.image_prompt or "").strip()
            if not prompt:
                raise ValueError("请上传人像，或填写 Flux 出图提示词")
            log("info", "没有底图，Flux 先出一张竖屏人像")
            image_name = comfy_image.generate_image(
                prompt,
                prompt_en=prompt,
                negative=negative,
                width=FLUX_WIDTH,
                height=FLUX_HEIGHT,
                seed=req.seed,
                filename_prefix="s2v_flux",
                log_callback=log,
            )
        tasks[tid]["progress"] = 25

        audio_name = (req.audio_name or "").strip()
        duration_s = 0.0
        if audio_name:
            audio_path = UPLOAD_DIR / audio_name
            if not audio_path.exists():
                raise FileNotFoundError(f"音频不存在: {audio_name}")
            duration_s = _probe_audio_duration(audio_path)
            log("info", "使用已上传音频", {"audio": audio_name, "duration_s": duration_s})
        else:
            dialogue = (req.dialogue or "").strip()
            if not dialogue:
                raise ValueError("请填写台词，或上传一段音频")
            log("info", "开始合成配音")
            audio_name, duration_s = tts.synthesize(dialogue, voice=req.voice or DEFAULT_VOICE)
            log("info", "配音完成", {"audio": audio_name, "duration_s": duration_s})
        if duration_s <= 0:
            duration_s = 3.0
        duration_s = min(5.0, max(1.0, float(duration_s)))
        tasks[tid]["progress"] = 40
        log("info", f"开始 Wan S2V 对口型，约 {duration_s:.1f} 秒")

        res = video_gen.generate_shot_video(
            image_name=image_name,
            audio_name=audio_name,
            positive_prompt=motion,
            negative_prompt=negative,
            width=1088,
            height=1920,
            duration_s=duration_s,
            seed=req.seed,
            mode="s2v",
            quality="fast",
            max_len=81,
            log_callback=log,
        )
        video_name = res["filename"]
        local_dir = ROOT / "output"
        local_dir.mkdir(parents=True, exist_ok=True)
        src = OUTPUT_DIR / video_name
        if src.exists():
            shutil.copy2(src, local_dir / video_name)
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {video_name}"
        tasks[tid]["results"] = [{"image": image_name, "audio": audio_name, "video": video_name, "duration_s": duration_s}]
        log("info", "S2V 完成", {"image": image_name, "audio": audio_name, "video": video_name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"S2V 失败: {e}"
        _task_log(tid, "error", str(e))


class TalkingReq(BaseModel):
    image_name: str = ""
    image_prompt: str = ""
    audio_name: str = ""
    dialogue: str = ""
    voice: str = DEFAULT_VOICE


def _run_talking(tid: str, req: TalkingReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "准备快速数字人", "results": [], "logs": []}

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        image_name = (req.image_name or "").strip()
        if not image_name:
            prompt = (req.image_prompt or "").strip()
            if not prompt:
                raise ValueError("请上传人像，或填写 Flux 出图提示词")
            log("info", "没有底图，Flux 先出一张竖屏人像")
            image_name = comfy_image.generate_image(
                prompt,
                prompt_en=prompt,
                negative="",
                width=FLUX_WIDTH,
                height=FLUX_HEIGHT,
                seed=None,
                filename_prefix="talking_flux",
                log_callback=log,
            )
        tasks[tid]["progress"] = 25

        audio_name = (req.audio_name or "").strip()
        duration_s = 0.0
        if audio_name:
            audio_path = UPLOAD_DIR / audio_name
            if not audio_path.exists():
                raise FileNotFoundError(f"音频不存在: {audio_name}")
            duration_s = _probe_audio_duration(audio_path)
            log("info", "使用已上传音频", {"audio": audio_name, "duration_s": duration_s})
        else:
            dialogue = (req.dialogue or "").strip()
            if not dialogue:
                raise ValueError("请填写台词，或上传一段音频")
            log("info", "开始合成配音")
            audio_name, duration_s = tts.synthesize(dialogue, voice=req.voice or DEFAULT_VOICE)
            log("info", "配音完成", {"audio": audio_name, "duration_s": duration_s})
        tasks[tid]["progress"] = 45
        log("info", "开始照片口型驱动（不跑 14B）")
        res = talking_fast.generate_talking_photo(image_name, audio_name, log_callback=log)
        video_name = res["filename"]
        duration_s = float(res.get("duration_s") or duration_s or 0)
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {video_name}"
        tasks[tid]["results"] = [{"image": image_name, "audio": audio_name, "video": video_name, "duration_s": duration_s, "backend": "photo_drive"}]
        log("info", "快速数字人完成", {"image": image_name, "audio": audio_name, "video": video_name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"快速数字人失败: {e}"
        _task_log(tid, "error", str(e))


@app.post("/api/talking/generate")
def api_talking_generate(req: TalkingReq) -> dict[str, Any]:
    if not (req.image_name or "").strip() and not (req.image_prompt or "").strip():
        raise HTTPException(400, "请上传人像，或填写出图提示词")
    if not (req.audio_name or "").strip() and not (req.dialogue or "").strip():
        raise HTTPException(400, "请填写台词，或上传音频")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_talking, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


class TalkingVideoReq(BaseModel):
    video_name: str = ""
    audio_name: str = ""
    dialogue: str = ""
    voice: str = DEFAULT_VOICE


def _run_talking_video(tid: str, req: TalkingVideoReq) -> None:
    tasks[tid] = {"status": "running", "progress": 5, "message": "准备视频数字人", "results": [], "logs": []}

    def log(level: str, msg: str, data: dict[str, Any] | None = None) -> None:
        _task_log(tid, level, msg, data)
        task = tasks.get(tid)
        if task and task.get("status") == "running":
            task["message"] = msg

    try:
        video_name = (req.video_name or "").strip()
        if not video_name:
            raise ValueError("请先上传一段正脸视频")
        video_path = UPLOAD_DIR / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_name}")
        tasks[tid]["progress"] = 20

        audio_name = (req.audio_name or "").strip()
        duration_s = 0.0
        if audio_name:
            audio_path = UPLOAD_DIR / audio_name
            if not audio_path.exists():
                raise FileNotFoundError(f"音频不存在: {audio_name}")
            duration_s = _probe_audio_duration(audio_path)
            log("info", "使用已上传音频", {"audio": audio_name, "duration_s": duration_s})
        elif (req.dialogue or "").strip():
            log("info", "开始合成配音")
            audio_name, duration_s = tts.synthesize((req.dialogue or "").strip(), voice=req.voice or DEFAULT_VOICE)
            log("info", "配音完成", {"audio": audio_name, "duration_s": duration_s})
        else:
            log("info", "没有新台词，抽取视频原声")
            audio_name, duration_s = _extract_audio_from_video(video_path)
            log("info", "已用视频原声", {"audio": audio_name, "duration_s": duration_s})
        tasks[tid]["progress"] = 45
        log("info", "开始视频口型驱动（不跑 14B）")
        res = talking_fast.generate_talking_video(video_name, audio_name, log_callback=log)
        video_out = res["filename"]
        duration_s = float(res.get("duration_s") or duration_s or 0)
        still = str(res.get("image_name") or "")
        tasks[tid]["status"] = "done"
        tasks[tid]["progress"] = 100
        tasks[tid]["message"] = f"完成 {video_out}"
        tasks[tid]["results"] = [{
            "image": still,
            "audio": audio_name,
            "video": video_out,
            "duration_s": duration_s,
            "backend": "video_drive",
            "source_video": video_name,
        }]
        log("info", "视频数字人完成", {"video": video_out, "audio": audio_name})
    except Exception as e:
        tasks[tid]["status"] = "error"
        tasks[tid]["progress"] = 0
        tasks[tid]["message"] = f"视频数字人失败: {e}"
        _task_log(tid, "error", str(e))


@app.post("/api/talking-video/generate")
def api_talking_video_generate(req: TalkingVideoReq) -> dict[str, Any]:
    if not (req.video_name or "").strip():
        raise HTTPException(400, "请先上传一段视频")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_talking_video, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


@app.post("/api/s2v/generate")
def api_s2v_generate(req: S2VReq) -> dict[str, Any]:
    if not (req.image_name or "").strip() and not (req.image_prompt or "").strip():
        raise HTTPException(400, "请上传人像，或填写出图提示词")
    if not (req.audio_name or "").strip() and not (req.dialogue or "").strip():
        raise HTTPException(400, "请填写台词，或上传音频")
    tid = _random_id()
    tasks[tid] = {"status": "queued", "progress": 0, "message": "排队中", "results": [], "logs": []}
    import threading

    threading.Thread(target=_run_s2v, args=(tid, req), daemon=True).start()
    return {"task_id": tid, "status": "queued"}


@app.get("/api/generate/status/{tid}")
def api_generate_status(tid: str) -> dict[str, Any]:
    return tasks.get(tid, {"status": "unknown", "progress": 0, "message": "任务不存在", "results": [], "logs": []})


@app.get("/comfy-output/{filename:path}")
def api_comfy_output(filename: str, download: bool = False) -> FileResponse:
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    if download:
        return FileResponse(path, filename=path.name)
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CONSOLE_HOST, port=CONSOLE_PORT)
