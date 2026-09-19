# coding=utf-8
"""TTS generation via the local Qwen3-TTS Gradio service (port 6009)."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from gradio_client import Client

from config import TTS_URL, UPLOAD_DIR


_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(str(TTS_URL))
    return _client


def _reset_client() -> None:
    global _client
    _client = None


def synthesize(dialogue: str, voice: str = "01 温柔女声·播音腔", sample_rate: int = 44100) -> tuple[str, float]:
    """Generate a single audio file from dialogue. Returns (audio_name, duration_s)."""
    if not dialogue.strip():
        raise ValueError("Dialogue is empty")

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            client = _get_client()
            result = client.predict(
                dialogue,
                voice,
                str(sample_rate),
                api_name="/generate",
            )
            break
        except Exception as e:
            last_err = e
            _reset_client()
            if attempt == 0:
                continue
            raise RuntimeError(
                f"TTS 服务连不上（{TTS_URL}）。如果正在使用 s2v_console_v2/start.sh，它会自动拉起 TTS；"
                f"否则请手动启动 /root/ComfyUI/ttsvoice/start_autodl.sh。原始错误: {e}"
            ) from e
    else:
        raise last_err or RuntimeError("TTS failed")
    # result: (audio_path_or_url, status_text)
    if not result or not isinstance(result, tuple):
        raise RuntimeError(f"Unexpected TTS result: {result}")

    audio_path = result[0]
    if not audio_path:
        raise RuntimeError(f"TTS returned no audio path: {result}")

    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"TTS output not found: {src}")

    suffix = src.suffix or ".wav"
    audio_name = f"console_tts_{random.randint(100000, 999999)}{suffix}"
    dest = UPLOAD_DIR / audio_name
    shutil.copy2(src, dest)

    duration_s = _audio_duration(dest)
    return audio_name, duration_s


def _audio_duration(path: Path) -> float:
    try:
        import wave

        with wave.open(str(path), "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0
