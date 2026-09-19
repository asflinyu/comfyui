# coding=utf-8
"""Customer TTS studio: single line + storyboard batch (video_clip wavs)."""
from __future__ import annotations

import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DEPS = _ROOT / ".deps"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("QWEN_TTS_OUTPUT_SR", "24000")

import librosa
import numpy as np
import soundfile as sf
import torch
import gradio as gr
from qwen_tts import Qwen3TTSModel

from script_parse import Clip, clips_table, parse_script
from voices import PREVIEW_TEXT, get_voice, voice_choices

PREVIEW_DIR = _ROOT / "voice_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR = _ROOT / "batch_outputs"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATES = [24000, 44100, 48000]

EXAMPLE_SCRIPT = """视频1，0-3s：产品拍了几十条，播放量还是几百。 3-6s：剪到半夜，点赞还是个位数。 6-9s：老板催着要爆款，运营剪到怀疑人生。 9-12s：三百播放，五个赞，白忙活一场。 12-15s：卖家做短视频的崩溃，谁懂啊。
视频2：0-3s：视频一直没流量，老板愁到挠头。 3-6s：运营没有放弃，换了 AI 做视频。 6-9s：突然爆了，播放量疯狂上涨。 9-12s：完播率 65%，转化率冲到 12%。 12-15s：找对方法，短视频也能逆风翻盘。"""


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> tuple[np.ndarray, int]:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    src_sr, dst_sr = int(src_sr), int(dst_sr)
    if src_sr != dst_sr:
        wav = librosa.resample(y=wav, orig_sr=src_sr, target_sr=dst_sr)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32), dst_sr


def _read_upload(file_obj) -> str:
    if file_obj is None:
        return ""
    path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_model(ckpt: str) -> Qwen3TTSModel:
    return Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


def build_app(tts: Qwen3TTSModel) -> gr.Blocks:
    def synthesize(text: str, instruct: str):
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language="Auto",
            instruct=instruct.strip(),
        )
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)

    def preview(voice_name: str, sample_rate: int):
        voice = get_voice(voice_name)
        cache = PREVIEW_DIR / f"{voice['id']}.wav"
        try:
            if cache.exists() and cache.stat().st_size > 1000:
                wav, sr = sf.read(str(cache), dtype="float32")
            else:
                wav, sr = synthesize(PREVIEW_TEXT, voice["instruct"])
                sf.write(str(cache), wav, sr)
            out, out_sr = _resample(wav, sr, int(sample_rate))
            return (out_sr, out), f"试听完成：{voice['name']}  |  {out_sr} Hz\n{voice['instruct']}"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def generate(text: str, voice_name: str, sample_rate: int):
        if not (text or "").strip():
            return None, "请填写要合成的文本。"
        voice = get_voice(voice_name)
        try:
            wav, sr = synthesize(text.strip(), voice["instruct"])
            out, out_sr = _resample(wav, sr, int(sample_rate))
            return (out_sr, out), f"生成完成：{voice['name']}  |  模型原生 24 kHz → 输出 {out_sr} Hz"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def resolve_script(raw: str, upload) -> str:
        uploaded = _read_upload(upload).strip()
        typed = (raw or "").strip()
        return uploaded or typed

    def parse_only(raw: str, upload):
        script = resolve_script(raw, upload)
        clips = parse_script(script)
        if not clips:
            return [], "没有解析到分镜。请使用「视频1」「0-3s：」这种格式。"
        n_vid = len({c.video_index for c in clips})
        return clips_table(clips), f"解析到 {n_vid} 个视频，共 {len(clips)} 条分镜音频。"

    def generate_batch(raw: str, upload, voice_name: str, sample_rate: int, progress=gr.Progress()):
        script = resolve_script(raw, upload)
        clips = parse_script(script)
        if not clips:
            return [], None, None, "没有解析到分镜。"
        voice = get_voice(voice_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = BATCH_DIR / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        last_audio = None
        try:
            for i, clip in enumerate(clips):
                progress((i + 1) / len(clips), desc=f"生成 {clip.filename}")
                dur = clip.duration_s
                instruct = (
                    f"{voice['instruct']} 请把下面这句话在大约{dur:g}秒内自然说完，"
                    "语速据此调整，不要抢戏也不要拖太久。"
                )
                wav, sr = synthesize(clip.text, instruct)
                out, out_sr = _resample(wav, sr, int(sample_rate))
                path = out_dir / clip.filename
                sf.write(str(path), out, out_sr)
                last_audio = (out_sr, out)
            zip_path = out_dir / "storyboard_wavs.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for clip in clips:
                    zf.write(out_dir / clip.filename, arcname=clip.filename)
            n_vid = len({c.video_index for c in clips})
            names = ", ".join(c.filename for c in clips)
            msg = (
                f"完成：{n_vid} 个视频，{len(clips)} 条音频，{int(sample_rate)} Hz\n"
                f"命名：视频号_镜号.wav，例如 1_1.wav、2_3.wav\n"
                f"目录：{out_dir}\n{names}"
            )
            return clips_table(clips), last_audio, str(zip_path), msg
        except Exception as e:
            return clips_table(clips), None, None, f"{type(e).__name__}: {e}"

    names = voice_choices()
    default = names[0]
    with gr.Blocks(title="Qwen3-TTS 配音工作台", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Qwen3-TTS 配音工作台")
        with gr.Tabs():
            with gr.Tab("分镜批量配音"):
                gr.Markdown(
                    """
上传或粘贴分镜文案。会按 **视频** 拆开，再按时间轴拆成多条小音频：

- 视频1 → `1_1.wav` `1_2.wav` …
- 视频2 → `2_1.wav` `2_2.wav` …

格式示例：`视频1，0-3s：…… 3-6s：……` 后面可以继续 `视频2：0-3s：……`
"""
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        file_in = gr.File(label="上传文案（txt）", file_types=[".txt", ".md"])
                        script_in = gr.Textbox(
                            label="或直接粘贴文案",
                            lines=10,
                            value=EXAMPLE_SCRIPT,
                        )
                        voice_b = gr.Dropdown(choices=names, value=default, label="音色")
                        sr_b = gr.Radio(choices=SAMPLE_RATES, value=44100, label="输出采样率（Hz）")
                        with gr.Row():
                            parse_btn = gr.Button("只解析，不生成")
                            batch_btn = gr.Button("解析并生成全部音频", variant="primary")
                    with gr.Column(scale=3):
                        table = gr.Dataframe(
                            headers=["文件名", "视频", "时间", "时长", "文案"],
                            label="解析结果",
                            wrap=True,
                        )
                        last_out = gr.Audio(label="最后一条试听", type="numpy")
                        zip_out = gr.File(label="打包下载全部 wav")
                        batch_status = gr.Textbox(label="状态", lines=6)
                parse_btn.click(parse_only, inputs=[script_in, file_in], outputs=[table, batch_status])
                batch_btn.click(
                    generate_batch,
                    inputs=[script_in, file_in, voice_b, sr_b],
                    outputs=[table, last_out, zip_out, batch_status],
                )

            with gr.Tab("单条配音"):
                gr.Markdown("没有参考音频时，从 50 种音色里试听后再合成。模型原生 24 kHz，可导出 44.1k / 48k。")
                with gr.Row():
                    with gr.Column(scale=2):
                        voice_dd = gr.Dropdown(choices=names, value=default, label="选择音色（50种）")
                        voice_desc = gr.Textbox(
                            label="音色特征",
                            value=get_voice(default)["instruct"],
                            lines=3,
                            interactive=False,
                        )
                        sr_in = gr.Radio(choices=SAMPLE_RATES, value=44100, label="输出采样率（Hz）")
                        preview_btn = gr.Button("试听该音色")
                        preview_audio = gr.Audio(label="试听", type="numpy")
                        text_in = gr.Textbox(label="要合成的文本", lines=5, value="欢迎使用配音工作台。")
                        gen_btn = gr.Button("生成配音", variant="primary")
                    with gr.Column(scale=3):
                        audio_out = gr.Audio(label="合成结果（可下载）", type="numpy")
                        status = gr.Textbox(label="状态", lines=4)
                voice_dd.change(lambda name: get_voice(name)["instruct"], inputs=voice_dd, outputs=voice_desc)
                preview_btn.click(preview, inputs=[voice_dd, sr_in], outputs=[preview_audio, status])
                gen_btn.click(generate, inputs=[text_in, voice_dd, sr_in], outputs=[audio_out, status])
    return demo


def main() -> None:
    ckpt = os.environ.get(
        "QWEN_TTS_CKPT",
        str(_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
    ip = os.environ.get("QWEN_TTS_IP", "127.0.0.1")
    port = int(os.environ.get("QWEN_TTS_PORT", "7860"))
    print(f"[studio] loading {ckpt}", flush=True)
    tts = load_model(ckpt)
    print("[studio] model ready", flush=True)
    app = build_app(tts)
    app.queue(default_concurrency_limit=1).launch(server_name=ip, server_port=port, share=False)


if __name__ == "__main__":
    main()
