#!/usr/bin/env python3
"""一键跑完全程的端到端测试脚本（最小成本版）。"""
from __future__ import annotations

import json
import sys
import time
from typing import Any

import httpx

BASE = "http://127.0.0.1:6010"


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
        print(f"[{label}] {d.get('status')} {d.get('progress', 0)}% - {d.get('message')} (task_id={tid})")
        logs = d.get("logs") or []
        for entry in logs[last_logs:]:
            ts = entry.get("ts", "")
            level = entry.get("level", "info")
            msg = entry.get("message", "")
            data = entry.get("data")
            data_str = f" {json.dumps(data, ensure_ascii=False)}" if data else ""
            print(f"  [{ts}] {level} | {msg}{data_str}")
        last_logs = len(logs)
        if d.get("status") == "done":
            return d
        if d.get("status") == "error":
            raise RuntimeError(f"{label} failed: {d.get('message')}")
        time.sleep(2)
    raise TimeoutError(f"{label} timed out")


def main() -> int:
    brief = "一只可爱的小猫在阳光明媚的草地上追逐蝴蝶，适合短视频"
    print("=" * 60)
    print("开始一键跑完全程测试")
    print(f"需求: {brief}")

    # Step 1: storyboard
    print("\n[Step 1/4] 生成分镜...")
    story = post(
        "/api/storyboard",
        {
            "brief": brief,
            "target_seconds": 6,
            "preferred_shots": 3,
            "mode": "i2v",
            "project": None,
            "revise": False,
        },
        timeout=300,
    )
    print(f"标题: {story.get('title')}")
    print(f"镜头数: {len(story.get('shots', []))}")
    for s in story.get("shots", []):
        print(f"  镜头{s.get('index')}: {s.get('duration_s')}s | {s.get('dialogue')[:30]}...")

    # Step 2: images
    print("\n[Step 2/4] 生成参考图...")
    img_task = post("/api/images/generate_all", {"project": story, "width": 256, "height": 256})
    img_result = poll_task(img_task["task_id"], "生图", timeout_s=600)
    project = img_result.get("project") or story
    print(f"生图结果: {img_result.get('message')}")
    if img_result.get("errors"):
        print(f"  错误: {img_result['errors']}")

    # Step 3: tts
    print("\n[Step 3/4] 合成配音...")
    tts_task = post("/api/tts/all", {"project": project, "voice": "01 温柔女声·播音腔", "skip_existing": False})
    tts_result = poll_task(tts_task["task_id"], "TTS", timeout_s=300)
    project = tts_result.get("project") or project
    print(f"TTS结果: {tts_result.get('message')}")
    if tts_result.get("errors"):
        print(f"  错误: {tts_result['errors']}")

    # Step 4: videos
    print("\n[Step 4/4] 生成视频...")
    vid_task = post(
        "/api/generate/videos",
        {
            "project": project,
            "resolution": "smoke_128",  # 最小分辨率，最快
            "video_count": 1,
            "seed_base": 42,
            "voice": "01 温柔女声·播音腔",
            "skip_tts_if_audio": True,
            "mode": "i2v",
        },
    )
    vid_result = poll_task(vid_task["task_id"], "视频", timeout_s=600)
    print(f"视频结果: {vid_result.get('message')}")

    results = vid_result.get("results", [])
    videos = []
    for r in results:
        videos.extend(r.get("videos", []))

    if videos:
        print(f"\n✅ 全程完成，生成视频文件:")
        for v in videos:
            print(f"  - {v.get('filename')} -> {BASE}/comfy-output/{v.get('filename')}")
    else:
        print("\n⚠️ 没有生成视频文件")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
