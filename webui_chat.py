#!/usr/bin/env python3
"""Bonsai-8B + Irodori-TTS Web Chat with voice output."""
from __future__ import annotations

import os
import re
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gradio as gr
import numpy as np
import torch

# ---------------------------------------------------------------------------
# LLM (Bonsai-8B)
# ---------------------------------------------------------------------------

_llm_model = None
_llm_tokenizer = None
_llm_history: list[dict] = []
_llm_backend: str = ""  # "mlx" or "llama_cpp"

SYSTEM_PROMPT = "あなたは「みどり」という名前の親切で明るいAIアシスタントです。日本語で短く自然に会話してください。長くても3文以内で答えてください。絵文字は使わないでください。"

DEFAULT_CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"

def load_llm():
    global _llm_model, _llm_tokenizer, _llm_history, _llm_backend
    if _llm_model is not None:
        return

    llm_api_base = os.environ.get("LLM_API_BASE")
    if llm_api_base:
        from openai import OpenAI

        print(f"[llm] Using llama.cpp server at {llm_api_base}...")
        _llm_model = OpenAI(base_url=f"{llm_api_base}/v1", api_key="no-key")
        _llm_tokenizer = None
        _llm_backend = "llama_server"
    else:
        from mlx_lm import load

        print("[llm] Loading Bonsai-8B 1-bit via MLX...")
        _llm_model, _llm_tokenizer = load("prism-ml/Bonsai-8B-mlx-1bit")
        _llm_backend = "mlx"

    _llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"[llm] Bonsai-8B loaded! (backend={_llm_backend})")


def llm_respond(message: str, history: list[dict]) -> str:
    global _llm_history

    _llm_history.append({"role": "user", "content": message})

    if _llm_backend == "llama_server":
        result = _llm_model.chat.completions.create(
            model="bonsai-8b",
            messages=_llm_history,
            max_tokens=256,
        )
        msg = result.choices[0].message
        response = msg.content or getattr(msg, "reasoning_content", "") or ""
    else:
        from mlx_lm import generate

        prompt = _llm_tokenizer.apply_chat_template(
            _llm_history, tokenize=False, add_generation_prompt=True
        )
        response = generate(_llm_model, _llm_tokenizer, prompt=prompt, max_tokens=256)

    clean = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    clean = re.sub(r"</?think>", "", clean).strip()
    _llm_history.append({"role": "assistant", "content": clean})
    return clean


# ---------------------------------------------------------------------------
# TTS (Irodori-TTS VoiceDesign)
# ---------------------------------------------------------------------------

_tts_runtime = None


def load_tts():
    global _tts_runtime
    if _tts_runtime is not None:
        return
    from irodori_tts.inference_runtime import (
        RuntimeKey,
        get_cached_runtime,
        default_runtime_device,
    )
    from huggingface_hub import hf_hub_download

    print("[tts] Loading Irodori-TTS VoiceDesign...")
    checkpoint_path = hf_hub_download(
        repo_id="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        filename="model.safetensors",
    )
    device = default_runtime_device()
    precision = "bf16" if str(device) == "cuda" else "fp32"
    key = RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=device,
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=precision,
        codec_device=device,
        codec_precision="fp32",
        enable_watermark=False,
        compile_model=False,
        compile_dynamic=False,
    )
    _tts_runtime, _ = get_cached_runtime(key)
    print(f"[tts] Irodori-TTS loaded! (device={device}, precision={precision})")


def tts_synthesize(text: str, caption: str, num_steps: int) -> tuple[tuple[int, np.ndarray], str]:
    from irodori_tts.inference_runtime import SamplingRequest

    t0 = time.time()
    result = _tts_runtime.synthesize(
        SamplingRequest(
            text=text, caption=caption,
            ref_wav=None, ref_latent=None, no_ref=True,
            ref_normalize_db=-16.0, ref_ensure_max=True,
            num_candidates=1, decode_mode="sequential",
            seconds=30.0, max_ref_seconds=30.0, max_text_len=None,
            num_steps=num_steps, seed=None,
            cfg_guidance_mode="independent",
            cfg_scale_text=2.0, cfg_scale_caption=4.0, cfg_scale_speaker=0.0,
            cfg_scale=None, cfg_min_t=0.5, cfg_max_t=1.0,
            truncation_factor=None, rescale_k=None, rescale_sigma=None,
            context_kv_cache=True, speaker_kv_scale=None,
            speaker_kv_min_t=None, speaker_kv_max_layers=None,
            trim_tail=True,
        ),
    )
    tts_time = time.time() - t0
    audio = result.audios[0].squeeze(0).float().numpy()
    sr = result.sample_rate
    return (sr, audio), f"TTS: {tts_time:.1f}s"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    print("[ui] Loading models at startup...")
    load_llm()
    load_tts()
    print("[ui] All models loaded!")

    with gr.Blocks(title="Midori Chat - Bonsai + Irodori-TTS") as demo:
        backend_label = "llama.cpp 1-bit CUDA" if _llm_backend == "llama_server" else "1-bit MLX"
        gr.Markdown(f"# Midori Chat\nBonsai-8B (1-bit {backend_label}) + Irodori-TTS VoiceDesign")

        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat", height=400)
            with gr.Row():
                msg = gr.Textbox(label="Message", placeholder="みどりに話しかけてね...", scale=4)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                audio_out = gr.Audio(label="Voice", autoplay=True)
                timing = gr.Textbox(label="Timing", interactive=False)

            with gr.Accordion("Settings", open=False):
                caption = gr.Textbox(label="Voice Caption", value=DEFAULT_CAPTION, lines=2)
                num_steps = gr.Slider(label="TTS Steps", minimum=5, maximum=40, value=15, step=1)
                auto_tts = gr.Checkbox(label="Auto TTS", value=True)

            def respond(message, chat_history, voice_caption, steps, do_tts):
                if not message.strip():
                    return "", chat_history, None, ""

                t0 = time.time()
                bot_response = llm_respond(message, chat_history)
                llm_time = time.time() - t0
                llm_info = f"LLM: {llm_time:.1f}s"

                chat_history = chat_history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": bot_response},
                ]

                audio = None
                info = llm_info
                if do_tts:
                    audio, tts_info = tts_synthesize(bot_response, voice_caption, int(steps))
                    info = f"{llm_info} | {tts_info}"

                return "", chat_history, audio, info

            send_btn.click(
                respond,
                inputs=[msg, chatbot, caption, num_steps, auto_tts],
                outputs=[msg, chatbot, audio_out, timing],
            )
            msg.submit(
                respond,
                inputs=[msg, chatbot, caption, num_steps, auto_tts],
                outputs=[msg, chatbot, audio_out, timing],
            )

        with gr.Tab("TTS Only"):
            gr.Markdown("テキストだけ音声化したい場合はこっち")
            tts_text = gr.Textbox(label="Text", lines=3)
            gr.Examples(
                examples=[
                    ["お電話ありがとうございます。ただいま電話が大変混み合っております。恐れ入りますが、発信音のあとに、ご用件をお話しください。"],
                    ["その森には、古い言い伝えがありました。月が最も高く昇る夜、静かに耳を澄ませば、風の歌声が聞こえるというのです。"],
                    ["なーに、どうしたの？…え？もっと近づいてほしい？…👂😮\u200d💨👂😮\u200d💨こういうのが好きなんだ？"],
                    ["うぅ…😭そんなに酷いこと、言わないで…😭"],
                    ["🤧🤧ごめんね、風邪引いちゃってて🤧…大丈夫、ただの風邪だからすぐ治るよ🥺"],
                ],
                inputs=[tts_text],
                label="Sample Texts",
            )
            tts_caption = gr.Textbox(label="Voice Caption", value=DEFAULT_CAPTION, lines=2)
            tts_steps = gr.Slider(label="TTS Steps", minimum=5, maximum=40, value=15, step=1)
            tts_btn = gr.Button("Generate Voice", variant="primary")
            tts_audio = gr.Audio(label="Generated Voice", autoplay=True)
            tts_timing = gr.Textbox(label="Timing", interactive=False)

            def tts_only(text, voice_caption, steps):
                if not text.strip():
                    return None, ""
                audio, info = tts_synthesize(text, voice_caption, int(steps))
                return audio, info

            tts_btn.click(
                tts_only,
                inputs=[tts_text, tts_caption, tts_steps],
                outputs=[tts_audio, tts_timing],
            )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(server_name=host, server_port=7860)
