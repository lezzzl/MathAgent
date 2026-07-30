#!/usr/bin/env bash
set -u

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Run ./install.sh first."
  exit 1
fi

source .venv/bin/activate

echo "== Build prerequisites =="
python - <<'PY'
import os
import sysconfig

include = sysconfig.get_path("include")
header = os.path.join(include, "Python.h")
print("Python include directory:", include)
print("Python.h:", header, "FOUND" if os.path.isfile(header) else "MISSING")
PY
command -v gcc || echo "gcc: MISSING"

echo
echo "== NVIDIA driver =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo
echo "== Python packages (without importing vLLM) =="
python - <<'PY'
import importlib.metadata
import sys

print("Python:", sys.version.replace("\n", " "))
for package in (
    "vllm",
    "torch",
    "transformers",
    "torchvision",
    "flashinfer-python",
    "causal-conv1d",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: not installed")
PY

echo
echo "== CUDA runtime libraries in the virtual environment =="
find .venv -type f -name 'libcudart.so*' -print 2>/dev/null || true

echo
echo "== vLLM extension dependencies =="
find .venv -type f \( -name '_C*.so' -o -name '_C*.abi3.so' \) -print0 2>/dev/null |
  while IFS= read -r -d '' extension; do
    echo "-- ${extension}"
    ldd "${extension}" 2>/dev/null | grep -E 'cudart|not found' || true
  done

echo
echo "== Import test =="
python - <<'PY'
import torch
print(f"PyTorch {torch.__version__}; build CUDA {torch.version.cuda}; available={torch.cuda.is_available()}")
import vllm
print(f"vLLM {vllm.__version__}: import OK")
PY
