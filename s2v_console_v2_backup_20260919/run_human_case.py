#!/usr/bin/env python3
"""跑一个真人主体完整案例：分镜 → 参考图 → 配音 → 1080×1920 视频。"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

BASE = "http://127.0.0.1:6010"
BRIEF = "一位年轻中国女孩在咖啡馆里收到好消息，开心地举杯庆祝。竖屏短视频，主体是真人，动作要连贯。"


def post(path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    r = httpx.post(f"{BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def poll_task(tid: str, label: str, timeout_s: int = 600) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_logs = 0
    while time.time() < deadline:
        r = httpx.get(f"{BASE}/api/generate/status/{tid}", timeout=30)
        r.raise_for_status()
        d = r.json()
        print(f"[{label}] {d.get('status')} {d.get('progress', 0)}% - {d.get('message')}")
        logs = d.get("logs") or []
        for entry in logs[last_logs:]:
            ts = entry.get("ts", "")
            level = entry.get("level", "info")
            msg = entry.get("message", "")
            data = entry.get("data")
            extra = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
            print(f"  [{ts}] {level} | {msg}{extra}")
        last_logs = len(logs)
        if d.get("status") == "done":
            return d
        if d.get("status") == "error":
            raise RuntimeError(f"{label} failed: {d.get('message')}")
        time.sleep(2)
    raise TimeoutError(f"{label} timed out")


def main() -> int:
    print("=" * 60)
    print("真人主体案例：咖啡馆好消息")
    print(f"需求: {BRIEF}")

    print("\n[Step 1/4] 生成分镜...")
    story = post(
        "/api/storyboard",
        {
            "brief": BRIEF,
            "target_seconds": 8,
            "preferred_shots": 4,
            "mode": "i2v",
            "project": None,
            "revise": False,
        },
        timeout=180,
    )
    print(f"标题: {story.get('title')}")
    print(f"梗概: {story.get('summary')}")
    for s in story.get("shots", []):
        print(f"\n镜头 {s.get('index')}  {s.get('duration_s')}s  运镜: {s.get('camera')}")
        print(f"  台词: {s.get('dialogue')}")
        print(f"  正向提示词: {s.get('positive_prompt')}")

    print("\n[Step 2/4] 生成参考图...")
    img_task = post("/api/images/generate_all", {"project": story, "width": 1080, "height": 1920})
    img_result = poll_task(img_task["task_id"], "生图")
    project = img_result.get("project") or story
    print(f"生图: {img_result.get('message')}")

    print("\n[Step 3/4] 合成配音...")
    tts_task = post(
        "/api/tts/all",
        {"project": project, "voice": "01 温柔女声·播音腔", "skip_existing": False},
    )
    tts_result = poll_task(tts_task["task_id"], "TTS")
    project = tts_result.get("project") or project
    print(f"TTS: {tts_result.get('message')}")

    print("\n[Step 4/4] 生成 1080×1920 视频...")
    vid_task = post(
        "/api/generate/videos",
        {
            "project": project,
            "resolution": "vertical_1080",
            "video_count": 1,
            "seed_base": 42,
            "voice": "01 温柔女声·播音腔",
            "skip_tts_if_audio": True,
            "mode": "i2v",
        },
    )
    vid_result = poll_task(vid_task["task_id"], "视频")
    print(f"视频: {vid_result.get('message')}")

    videos = []
    for r in vid_result.get("results") or []:
        videos.extend(r.get("videos") or [])

    print("\n" + "=" * 60)
    print("案例完成。下载 / 播放地址：")
    for v in videos:
        name = v.get("filename")
        if not name:
            continue
        print(f"  {BASE}/comfy-output/{name}")
        print(f"  下载: {BASE}/comfy-output/{name}?download=true")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
