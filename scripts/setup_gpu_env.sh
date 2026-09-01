#!/usr/bin/env bash
# 创建 GPU conda 环境并安装依赖
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${ENV_NAME:-basketball_classroom}"

echo "==> Project: $ROOT"
echo "==> Conda env: $ENV_NAME"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Please install Miniconda first."
  exit 1
fi

# 创建环境（若已存在则跳过）
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  echo "==> Env $ENV_NAME already exists"
else
  echo "==> Creating conda env (Python 3.10 + PyTorch CUDA 12.4)..."
  conda env create -f "$ROOT/environment.yml" -n "$ENV_NAME"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 注册 activate hook，使 onnxruntime-gpu 加载 libcudnn
HOOK_SRC="$ROOT/env/conda_activate.d/basketball_cuda.sh"
HOOK_DST="$CONDA_PREFIX/etc/conda/activate.d/basketball_cuda.sh"
mkdir -p "$(dirname "$HOOK_DST")"
cp "$HOOK_SRC" "$HOOK_DST"
# 当前 shell 立即生效
# shellcheck disable=SC1090
source "$HOOK_SRC"

echo "==> Installing cudnn (onnxruntime-gpu dependency)..."
conda install -y -c nvidia cudnn cuda-version=12 || true

echo "==> Installing pip dependencies..."
pip install -U pip
pip install -r "$ROOT/requirements.txt"
pip install -r "$ROOT/requirements-gpu.txt"
# 仅保留 GPU 版 onnxruntime（勿额外 pip install onnxruntime CPU 包）
pip uninstall -y onnxruntime 2>/dev/null || true

echo "==> GPU check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
try:
    import onnxruntime as ort
    print("onnxruntime:", ort.__version__)
    print("providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime:", e)
PY

echo ""
echo "==> Download models (may take several minutes)..."
python "$ROOT/scripts/download_models.py"

echo ""
echo "==> Verify installation"
python "$ROOT/scripts/verify_gpu_env.py"

echo ""
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
echo "  uvicorn apps.teacher_ui.main:app --host 127.0.0.1 --port 8000"
