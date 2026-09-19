#!/usr/bin/env python3
"""用 GEOschem.xlsx 跑快速版分镜全程。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

BASE = "http://127.0.0.1:6010"
XLSX = Path(__file__).resolve().parent / "GEOschem.xlsx"


def poll(tid: str, label: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last = 0
    while time.time() < deadline:
        d = httpx.get(f"{BASE}/api/generate/status/{tid}", timeout=30).json()
        print(f"[{label}] {d.get('status')} {d.get('progress', 0)}% - {d.get('message')}", flush=True)
        logs = d.get("logs") or []
        for entry in logs[last:]:
            data = entry.get("data")
            extra = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
            print(f"  {entry.get('level')} | {entry.get('message')}{extra}", flush=True)
        last = len(logs)
        if d.get("status") == "done":
            return d
        if d.get("status") == "error":
            raise RuntimeError(f"{label} failed: {d.get('message')}")
        time.sleep(2)
    raise TimeoutError(f"{label} timed out after {timeout_s}s")


def main() -> int:
    t0 = time.time()
    print("GEOschem.xlsx -> 快速版分镜", flush=True)
    with XLSX.open("rb") as f:
        r = httpx.post(
            f"{BASE}/api/storyboard/from_excel",
            files={"files": (XLSX.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"mode": "i2v"},
            timeout=60,
        )
    r.raise_for_status()
    project = r.json()
    shots = project.get("shots") or []
    print(f"标题: {project.get('title')}  镜头: {len(shots)}  id={project.get('id')}", flush=True)
    for s in shots:
        print(f"  {s['index']} {s.get('duration_s')}s | {s.get('dialogue')} | {s.get('visual')}", flush=True)

    print("\n[2/4] Qwen 4步参考图", flush=True)
    img = httpx.post(
        f"{BASE}/api/images/generate_all",
        json={"project": project, "width": 720, "height": 1280, "workflow": "qwen_t2i"},
        timeout=60,
    ).json()
    img_d = poll(img["task_id"], "生图", 1800)
    project = img_d.get("project") or project
    print("生图:", img_d.get("message"), img_d.get("errors") or "", flush=True)

    print("\n[3/4] TTS", flush=True)
    tts = httpx.post(
        f"{BASE}/api/tts/all",
        json={"project": project, "voice": "01 温柔女声·播音腔", "skip_existing": False},
        timeout=60,
    ).json()
    tts_d = poll(tts["task_id"], "TTS", 600)
    project = tts_d.get("project") or project
    print("TTS:", tts_d.get("message"), flush=True)

    print("\n[4/4] I2V fp8 4步 720p", flush=True)
    vid = httpx.post(
        f"{BASE}/api/generate/videos",
        json={
            "project": project,
            "resolution": "final_720",
            "quality": "fast",
            "video_count": 1,
            "skip_tts_if_audio": True,
            "mode": "i2v",
        },
        timeout=60,
    ).json()
    vid_d = poll(vid["task_id"], "视频", 5400)
    print("视频:", vid_d.get("message"), flush=True)

    videos = []
    for item in vid_d.get("results") or []:
        videos.extend(item.get("videos") or [])
    elapsed = time.time() - t0
    print(f"\n耗时 {elapsed/60:.1f} 分钟", flush=True)
    if not videos:
        print("没有成片", flush=True)
        return 1
    for v in videos:
        print(f"  {BASE}/comfy-output/{v.get('filename')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
