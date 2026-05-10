# Model Package Layout

TorchInferno follows the TorchTitan model-package shape where each model family
owns one directory and the runtime consumes that family through stable public
exports.

Canonical family packages:

- `dsv4/`: compact DeepSeek-style reference family.
- `deepseek_v32/`: native DeepSeek-V3.2 tensor-contract family.
- `llama3/`: Llama3 reference family plus current production-scale adapters.

Each family package should prefer this structure:

- `model.py`: pure readable single-device model code, config dataclasses,
  forward contracts, cache objects, and checkpoint-facing model class.
- `traceable_model.py`: the smallest adjustments needed for make_fx capture,
  such as no-cache full-prefix forward wrappers and stable sample inputs.
- `raw_ops.py`: provider-independent Python/PyTorch operation boundaries for
  the traceable reference.
- `fused_ops.py`: checked-in promoted operation hooks that preserve the raw
  contract.
- `v0.py`: make_fx-backed reference variant with `v0_graph()` and
  `print_readable()` helpers.
- `v1.py`, ...: provenance-tracked promoted variants.
- `registry.py`: variant parentage, class path, ops module, status, and notes.
- `pipeline.py` / `tensor_parallel.py`: optional family-specific execution
  adapters when a production-scale shape needs them.
- `state_dict_adapter.py`: optional future home for checkpoint mapping if a
  family-specific adapter grows beyond `models/conversion.py`.

New code should import from the canonical family packages. The older
`deepseek.py` module remains as a compatibility alias for DeepSeek-V3.2.

Model code should stay torch-native and runtime-ready. Offline graph capture,
partitioning, backend search, and promotion live outside the model hot path; see
`docs/OFFLINE_OPTIMIZATION.md`.
