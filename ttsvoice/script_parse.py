# coding=utf-8
"""Parse storyboard scripts into videos and timed clips."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class Clip:
    video_index: int
    clip_index: int
    start_s: float
    end_s: float
    text: str

    @property
    def filename(self) -> str:
        return f"{self.video_index}_{self.clip_index}.wav"

    @property
    def duration_s(self) -> float:
        return max(0.5, float(self.end_s) - float(self.start_s))


_VIDEO_SPLIT = re.compile(r"(?:^|[，,。；;\s])视频\s*(\d+)\s*[：:，,]?", re.M)
_SEG = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-~～到至]\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[：:]\s*",
    re.I,
)


def parse_script(raw: str) -> List[Clip]:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    marks = list(_VIDEO_SPLIT.finditer(text))
    if not marks:
        clips = _parse_segments(1, text)
        return clips

    clips: List[Clip] = []
    for i, m in enumerate(marks):
        vid = int(m.group(1))
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[start:end].strip(" ，,：:\n")
        clips.extend(_parse_segments(vid, body))
    return clips


def _parse_segments(video_index: int, body: str) -> List[Clip]:
    hits = list(_SEG.finditer(body))
    if not hits:
        line = body.strip()
        if not line:
            return []
        return [
            Clip(
                video_index=video_index,
                clip_index=1,
                start_s=0,
                end_s=3,
                text=line,
            )
        ]

    out: List[Clip] = []
    for i, m in enumerate(hits):
        t0 = float(m.group(1))
        t1 = float(m.group(2))
        s = m.end()
        e = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        piece = body[s:e].strip(" ，,。；;\n")
        if not piece:
            continue
        out.append(
            Clip(
                video_index=video_index,
                clip_index=len(out) + 1,
                start_s=t0,
                end_s=t1,
                text=piece,
            )
        )
    return out


def clips_table(clips: List[Clip]) -> list[list]:
    rows = []
    for c in clips:
        rows.append(
            [
                c.filename,
                f"视频{c.video_index}",
                f"{c.start_s:g}-{c.end_s:g}s",
                f"{c.duration_s:g}s",
                c.text,
            ]
        )
    return rows


def clips_to_dict(clips: List[Clip]) -> list[dict]:
    return [asdict(c) | {"filename": c.filename, "duration_s": c.duration_s} for c in clips]
