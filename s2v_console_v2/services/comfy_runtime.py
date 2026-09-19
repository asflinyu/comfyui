# coding=utf-8
"""Serialize Comfy GPU jobs. Keep VRAM for the same job type; unload on switch or idle."""
from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import httpx

from config import COMFY_URL

LogCb = Callable[[str, str, dict[str, Any] | None], None]
IDLE_FREE_S = 8.0


def _log(cb: LogCb | None, level: str, msg: str, data: dict[str, Any] | None = None) -> None:
    if callable(cb):
        cb(level, msg, data)


def gpu_used_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return -1.0


def comfy_busy() -> bool:
    try:
        q = httpx.get(f"{COMFY_URL}/queue", timeout=5).json()
        return bool(q.get("queue_running") or q.get("queue_pending"))
    except Exception:
        return False


def free_comfy_resources(log_callback: LogCb | None = None, wait: bool = True) -> None:
    before = gpu_used_mb()
    busy = comfy_busy()
    try:
        r = httpx.post(
            f"{COMFY_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        _log(log_callback, "warn", f"请求释放显存失败: {e}", {"vram_mb": before})
        return

    if busy:
        _log(log_callback, "info", "队列还在跑，跑完后会卸载模型", {"vram_mb": before})
        return
    if not wait:
        _log(log_callback, "info", "已请求 Comfy 卸载模型", {"vram_mb": before})
        return

    after = before
    for _ in range(25):
        time.sleep(0.4)
        if comfy_busy():
            break
        after = gpu_used_mb()
        if after >= 0 and before >= 0 and after <= before - 512:
            break
        if after >= 0 and after < 8000:
            break
    _log(
        log_callback,
        "info",
        "已卸载 Comfy 模型并释放显存",
        {"vram_mb_before": before, "vram_mb_after": after},
    )


class GpuGate:
    """One GPU job at a time.

    - Same kind (image/image or video/video) and more people waiting: keep weights loaded.
    - Kind switch (Flux/SD ↔ Wan): unload before the next job.
    - Queue empty: unload after IDLE_FREE_S.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._owner: int | None = None
        self._depth = 0
        self._waiters = 0
        self._idle_token = 0
        self._kind: str | None = None
        self._last_kind: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._cv:
            return {
                "busy": self._owner is not None,
                "waiting": self._waiters,
                "queue_len": self._waiters + (1 if self._owner is not None else 0),
                "kind": self._kind or self._last_kind,
            }

    def acquire(self, log_callback: LogCb | None = None, kind: str = "video") -> None:
        me = threading.get_ident()
        switch_from: str | None = None
        with self._cv:
            self._idle_token += 1
            if self._owner == me:
                self._depth += 1
                return
            self._waiters += 1
            ahead = self._waiters - 1 + (1 if self._owner is not None else 0)
            if ahead > 0:
                _log(log_callback, "info", f"GPU 排队中，前面还有 {ahead} 个任务", {"ahead": ahead, "kind": kind})
            while self._owner is not None:
                self._cv.wait()
            self._waiters -= 1
            if self._last_kind and self._last_kind != kind:
                switch_from = self._last_kind
            self._owner = me
            self._depth = 1
            self._kind = kind
            _log(log_callback, "info", "轮到本任务占用 GPU", {"kind": kind})
        if switch_from:
            _log(log_callback, "info", f"从 {switch_from} 切到 {kind}，先卸载旧模型")
            free_comfy_resources(log_callback)

    def release(self, log_callback: LogCb | None = None) -> None:
        me = threading.get_ident()
        with self._cv:
            if self._owner != me:
                return
            self._depth -= 1
            if self._depth > 0:
                return
            self._last_kind = self._kind
            self._kind = None
            self._owner = None
            idle = self._waiters == 0
            token = self._idle_token
            self._cv.notify()
        if idle:
            threading.Thread(
                target=self._idle_free,
                args=(token, log_callback),
                daemon=True,
                name="comfy-idle-free",
            ).start()
        else:
            _log(log_callback, "info", "后面还有同类任务，模型先留在显存里")

    def _idle_free(self, token: int, log_callback: LogCb | None) -> None:
        time.sleep(IDLE_FREE_S)
        with self._cv:
            if token != self._idle_token or self._owner is not None or self._waiters > 0:
                return
            self._last_kind = None
        _log(log_callback, "info", f"队列已空 {int(IDLE_FREE_S)}s，释放 GPU 显存")
        free_comfy_resources(log_callback)

    @contextmanager
    def hold(self, log_callback: LogCb | None = None, kind: str = "video") -> Iterator[None]:
        self.acquire(log_callback, kind=kind)
        try:
            yield
        finally:
            self.release(log_callback)


GPU = GpuGate()
