#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8}"
NEW_TOKENS="${NEW_TOKENS:-8}"
VOCAB_SIZE="${VOCAB_SIZE:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-0}"
COMPILE="${COMPILE:-0}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}/src"
fi

if [[ "${DEVICE}" == cuda* ]]; then
  "${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible to PyTorch. Check your driver, CUDA runtime, "
        "PyTorch build, and CUDA_VISIBLE_DEVICES."
    )

print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
PY
fi

cmd=(
  "${PYTHON_BIN}" -m torchinferno.cli dsv4-smoke
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --prompt-tokens "${PROMPT_TOKENS}"
  --new-tokens "${NEW_TOKENS}"
  --vocab-size "${VOCAB_SIZE}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
)

case "${COMPILE}" in
  1|true|TRUE|yes|YES)
    cmd+=(--compile)
    ;;
esac

cd "${REPO_ROOT}"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
