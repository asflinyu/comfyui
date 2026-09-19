# coding=utf-8
"""Excel / 文案 → 分镜，不走 LLM。给快速版控制台用。"""
from __future__ import annotations

import io
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import openpyxl

from config import DEFAULT_NEGATIVE, MAX_SHOTS, MIN_SHOTS

HEADER_WORDS = {"#", "序号", "镜头", "时间", "画面", "台词", "反转"}
TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-~–到至]\s*(\d+(?:\.\d+)?)")
SHOT_LINE_RE = re.compile(
    r"镜头\s*(\d+)\s*[（(]([^）)]+)[）)]\s*：\s*画面：(.+?)(?:。\s*台词：(.+?))?(?:。\s*反转：(.+))?$"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rid() -> str:
    return uuid.uuid4().hex[:10]


def duration_from_time(text: str, fallback: float = 2.0) -> float:
    raw = (text or "").strip()
    m = TIME_RE.search(raw)
    if m:
        return max(1.0, round(float(m.group(2)) - float(m.group(1)), 1))
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if m:
        return max(1.0, float(m.group(1)))
    return fallback


def _character_lock(title: str) -> str:
    if "老王" in title:
        return (
            "同一位中年中国男人火锅店老板老王，短黑发，疲惫皱纹，油渍围裙，真实皮肤，"
            "the same middle-aged Chinese hotpot shop owner Lao Wang, short black hair, tired face, stained apron"
        )
    return "同一位中国主角贯穿全片，真实皮肤，photorealistic Chinese lead character, consistent face"


def _prompts(title: str, visual: str, twist: str, dialogue: str) -> tuple[str, str, str]:
    lock = _character_lock(title)
    still = (
        f"竖屏9:16写实电影静帧，{lock}。画面：{visual}。"
        + (f"关键反转：{twist}。" if twist else "")
        + "纪录片光，无字幕无水印无logo，cinematic still, no text"
    )
    motion = (
        f"cinematic vertical video, {lock}, action: {visual}"
        + (f", twist: {twist}" if twist else "")
        + ", natural body motion, slight camera move, photorealistic, no morphing"
    )
    image_cn = f"{visual}" + (f"，{twist}" if twist else "")
    return still, motion, image_cn


def _shot_dict(
    index: int,
    time_col: str,
    visual: str,
    dialogue: str,
    twist: str,
    title: str,
) -> dict[str, Any]:
    still, motion, image_cn = _prompts(title, visual, twist, dialogue)
    return {
        "index": index,
        "duration_s": duration_from_time(time_col),
        "time": time_col,
        "dialogue": dialogue,
        "positive_prompt": motion,
        "negative_prompt": DEFAULT_NEGATIVE,
        "image_prompt": still,
        "image_prompt_en": still,
        "camera": twist or "手持跟拍，动作清楚",
        "visual": visual,
        "twist": twist,
        "image_name": "",
        "image_source": "",
        "audio_name": "",
        "audio_duration_s": 0.0,
        "length": 0,
    }


def parse_excel_bytes(data: bytes) -> list[dict[str, Any]]:
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    scripts: list[dict[str, Any]] = []
    title = ""
    shots: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal title, shots
        if shots:
            scripts.append({"title": title or "未命名文案", "shots": shots})
        title, shots = "", []

    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            raw = ["" if c is None else str(c).strip() for c in row]
            cells = [c for c in raw if c]
            if not cells:
                continue
            first = cells[0]
            if all(c in HEADER_WORDS or c.replace(" ", "") in HEADER_WORDS for c in cells):
                continue
            if first in HEADER_WORDS and len(cells) <= 5 and any(c in HEADER_WORDS for c in cells[1:]):
                continue
            if len(cells) == 1:
                flush()
                title = first
                continue
            if len(cells) >= 4:
                idx_s, time_col, visual, dialogue = cells[0], cells[1], cells[2], cells[3]
                twist = cells[4] if len(cells) > 4 else ""
                try:
                    idx = int(re.sub(r"\D+", "", str(idx_s)) or len(shots) + 1)
                except ValueError:
                    idx = len(shots) + 1
                shots.append(_shot_dict(idx, time_col, visual, dialogue, twist, title))
    flush()
    return scripts


def parse_script_text(text: str) -> list[dict[str, Any]]:
    scripts: list[dict[str, Any]] = []
    title = ""
    shots: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal title, shots
        if shots:
            scripts.append({"title": title or "未命名文案", "shots": shots})
        title, shots = "", []

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SHOT_LINE_RE.match(line)
        if m:
            shots.append(
                _shot_dict(
                    int(m.group(1)),
                    m.group(2).strip(),
                    (m.group(3) or "").strip(),
                    (m.group(4) or "").strip(),
                    (m.group(5) or "").strip(),
                    title,
                )
            )
            continue
        if line.startswith("文案") or (not shots and "：" in line and "画面" not in line):
            flush()
            title = line
            continue
        if not shots:
            title = line
    flush()
    return scripts


def build_project(
    script: dict[str, Any],
    mode: str = "i2v",
    project: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shots = list(script.get("shots") or [])
    if len(shots) > MAX_SHOTS:
        shots = shots[:MAX_SHOTS]
    if len(shots) < MIN_SHOTS:
        raise ValueError(f"分镜至少 {MIN_SHOTS} 个镜头，当前 {len(shots)}")
    for i, shot in enumerate(shots):
        shot["index"] = i + 1
    title = script.get("title") or (project.get("title") if project else "快速分镜")
    title = re.sub(r"^文案\d+[：:]\s*", "", title)
    title = re.sub(r"\s*【.*?】\s*$", "", title).strip() or "快速分镜"
    return {
        "id": project.get("id") if project else _rid(),
        "title": title,
        "summary": f"快速版分镜，{len(shots)} 镜，不走 LLM",
        "mode": mode,
        "fast": True,
        "shots": shots,
        "updated_at": _now(),
    }
