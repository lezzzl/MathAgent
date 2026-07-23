#!/usr/bin/env bash
set -euo pipefail

# The unpinned vLLM nightly has had CUDA 12.9 wheels whose stable-libtorch
# extension is accidentally linked to libcudart.so.13. Keep the default on a
# known Qwen3.5-capable release. Override deliberately, for example:
#   VLLM_VERSION=0.25.1 ./install.sh
#   VLLM_CHANNEL=nightly ./install.sh
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
VLLM_CHANNEL="${VLLM_CHANNEL:-release}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.6}"

PYTHON_INCLUDE_DIR="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [[ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" ]]; then
  echo "ERROR: ${PYTHON_INCLUDE_DIR}/Python.h is missing." >&2
  echo "Triton needs the Python development headers to build its CUDA driver helper." >&2
  echo "On Ubuntu/Debian, run:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y python3.12-dev" >&2
  exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
  echo "ERROR: gcc is missing. On Ubuntu/Debian, install build-essential." >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv

if [[ "${VLLM_CHANNEL}" == "nightly" ]]; then
  uv pip install --upgrade vllm \
    --torch-backend="${TORCH_BACKEND}" \
    --extra-index-url "https://wheels.vllm.ai/nightly/${TORCH_BACKEND}"
else
  uv pip install "vllm==${VLLM_VERSION}" --torch-backend="${TORCH_BACKEND}"
fi

# A pre-existing nightly environment can retain a much newer Transformers
# package. vLLM 0.19.x carries its own Qwen3.5 config integration and is known
# to work with the 4.57 generation; Transformers 5.x changes the config types
# used during model inspection.
if [[ "${VLLM_CHANNEL}" == "release" && "${VLLM_VERSION}" == 0.19.* ]]; then
  uv pip install "transformers==${TRANSFORMERS_VERSION}"
fi

uv pip install -r requirements-client.txt

python - <<'PY'
import importlib.metadata
import torch

print(f"Python: {__import__('sys').version.split()[0]}")
print(f"PyTorch: {torch.__version__} (built for CUDA {torch.version.cuda})")
print(f"vLLM package: {importlib.metadata.version('vllm')}")
print(f"Transformers: {importlib.metadata.version('transformers')}")

import vllm
print(f"vLLM import: OK ({vllm.__version__})")
print(f"CUDA visible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
