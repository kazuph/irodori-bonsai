# irodori-bonsai

Local AI voice chat on Apple Silicon — [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) (1-bit LLM) + [Irodori-TTS VoiceDesign](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) (Japanese TTS).

Chat with "Midori", an AI assistant that talks back to you — entirely on your Mac, no cloud APIs needed.

## Features

- **Bonsai-8B** — 1-bit quantized 8B LLM via MLX, ~128 tok/s, 1.4GB memory
- **Irodori-TTS VoiceDesign** — Flow Matching Japanese TTS via PyTorch MPS, caption-controlled voice style
- **Web UI** — Gradio chat interface with auto voice playback
- **Fully local** — Both models run on Apple Silicon GPU (MLX + MPS)

## Performance (Mac M-series)

| Component | Speed | Memory |
|---|---|---|
| LLM (Bonsai-8B 1-bit) | 128 tok/s generation | 1.4 GB |
| TTS (15 steps) | ~13s for ~7s audio | ~4 GB |
| Total per turn | ~14s | ~5.4 GB |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12
- cmake (`brew install cmake`)
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

## Setup

```bash
# Install Metal Toolchain (one-time)
xcodebuild -downloadComponent MetalToolchain

# Run setup
bash setup_env.sh

# Activate and launch
source .venv/bin/activate
python webui_chat.py
```

Then open http://127.0.0.1:7860 in your browser.

## Usage

### Chat Tab
Type a message and press Send. Midori will respond in text and voice.

### TTS Only Tab
Enter any Japanese text to synthesize speech. Includes sample texts with emoji-based emotion control.

### Settings
- **Voice Caption** — Describe the voice style in Japanese (e.g., "落ち着いた女性の声で自然に読み上げてください")
- **TTS Steps** — 10-20 recommended (lower = faster, higher = better quality)
- **Auto TTS** — Toggle automatic voice synthesis

## Architecture

```
User Input → Bonsai-8B (MLX, Apple GPU) → Response Text
                                              ↓
                                    Irodori-TTS VoiceDesign (PyTorch MPS)
                                              ↓
                                    Audio Playback (browser autoplay)
```

## Credits

- [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit) by PrismML — 1-bit quantized LLM
- [Irodori-TTS](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) by Aratako — Flow Matching Japanese TTS
- [MLX](https://github.com/ml-explore/mlx) by Apple — ML framework for Apple Silicon
- [PrismML MLX fork](https://github.com/PrismML-Eng/mlx) — 1-bit kernel support

## License

MIT
