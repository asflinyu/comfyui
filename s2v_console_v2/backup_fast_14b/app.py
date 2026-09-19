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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from config import (
    COMFY_URL,
    CONSOLE_HOST,
    CONSOLE_PORT,
    DATA_DIR,
    DEFAULT_NEGATIVE,
    DEFAULT_VOICE,
    FPS,
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
    RESOLUTION_PRESETS,
    S2V_AUDIO_ENC,
    S2V_CLIP,
    S2V_LORA,
    S2V_UNET,
    S2V_VAE,
    SD_CKPT,
    TTS_URL,
    UPLOAD_DIR,
)
from services import comfy_image, deepseek, tts, video_gen

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


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="files")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


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
    }


@app.get("/api/voices")
def api_voices() -> dict[str, Any]:
    # TTS voice discovery is best-effort; fall back to default voice list
    voices = [DEFAULT_VOICE]
    try:
        r = httpx.get(f"{TTS_URL}/voices", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                voices = data
            elif isinstance(data, dict) and "voices" in data:
                voices = data["voices"]
    except Exception:
        pass
    return {"voices": voices}


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
                log_callback=lambda level, msg, data=None: _task_log(tid, level, msg, data),
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
