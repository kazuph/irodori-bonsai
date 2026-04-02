#!/bin/bash
# irodori-bonsai environment setup script
# Requires: Python 3.12, cmake, Metal Toolchain (xcodebuild -downloadComponent MetalToolchain)

set -e

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3.12}"

echo "=== irodori-bonsai setup ==="
echo "Python: $($PYTHON --version)"
echo "Venv: $VENV_DIR"

# Create venv
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Install PrismML fork of MLX (1-bit kernel support)
echo "=== Installing PrismML MLX (1-bit kernels) ==="
TEMP_DIR=$(mktemp -d)
git clone --branch prism --depth 1 https://github.com/PrismML-Eng/mlx.git "$TEMP_DIR/prism-mlx"

# Patch setup.py to use correct Python executable
cd "$TEMP_DIR/prism-mlx"
sed -i.bak "3a\\
import sys" setup.py
sed -i.bak 's|"-DBUILD_SHARED_LIBS=ON",|"-DBUILD_SHARED_LIBS=ON",\n            f"-DPython_EXECUTABLE={sys.executable}",|' setup.py
pip install --no-build-isolation --no-cache-dir .
cd -
rm -rf "$TEMP_DIR"

# Install mlx-lm (matching version)
echo "=== Installing mlx-lm ==="
pip install mlx-lm==0.30.7

# Install PyTorch + Irodori-TTS dependencies
echo "=== Installing PyTorch and TTS dependencies ==="
pip install torch torchaudio sounddevice textual
pip install "dacvae @ git+https://github.com/facebookresearch/dacvae" \
    soundfile peft gradio huggingface-hub safetensors \
    transformers sentencepiece pyyaml tqdm numba llvmlite

echo "=== Setup complete! ==="
echo "Run: source $VENV_DIR/bin/activate && python webui_chat.py"
