# Model Package Layout

TorchInferno follows the TorchTitan model-package shape where each model family
owns one directory and the runtime consumes that family through stable public
exports.

Canonical family packages:

- `dsv4/`: compact DeepSeek-style reference family.
- `deepseek_v32/`: native DeepSeek-V3.2 tensor-contract family.
- `llama3/`: Llama3 reference family plus current production-scale adapters.

Each family package should prefer this structure:

- `model.py`: readable single-device model code, config dataclasses, forward
  contracts, cache objects, and checkpoint-facing model class.
- `raw_ops.py`: provider-independent Python/PyTorch operation boundaries for
  the `v0` reference.
- `fused_ops.py`: checked-in promoted operation hooks that preserve the raw
  contract.
- `v0.py`, `v1.py`, ...: provenance-tracked variants.
- `registry.py`: variant parentage, class path, ops module, status, and notes.
- `pipeline.py` / `tensor_parallel.py`: optional family-specific execution
  adapters when a production-scale shape needs them.
- `state_dict_adapter.py`: optional future home for checkpoint mapping if a
  family-specific adapter grows beyond `models/conversion.py`.

Compatibility packages such as `dsv4_family/`, `deepseek_v32_family/`, and
`llama3_family/` remain as import shims. New code should import from the
canonical family packages.

Model code should stay torch-native and runtime-ready. Offline graph capture,
partitioning, backend search, and promotion live outside the model hot path; see
`docs/OFFLINE_OPTIMIZATION.md`.
