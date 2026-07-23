# Third-Party Notices

## DeepSeek-V4-Flash inference kernels

`src/torchinferno/kernels/deepseek_v4_tilelang_definitions.py` is adapted from the public
`deepseek-ai/DeepSeek-V4-Flash` inference reference, commit
`60d8d70770c6776ff598c94bb586a859a38244f1`.

Copyright (c) 2026 DeepSeek

Licensed under the MIT License. The complete required copyright, permission,
notice-retention, and warranty-disclaimer text is included in
`licenses/DeepSeek-V4-Flash-MIT.txt`.

## SGLang/vLLM CUDA utilities

`src/torchinferno/kernels/deepseek_v4_marlin.py` adapts Marlin weight-layout
and scale-layout algorithms from the SGLang and vLLM projects and invokes a
version-pinned internal SGLang Marlin provider. TorchInferno requires its
content-addressed shared library to be built by the explicit offline prepare
step before runtime model loading.

The mHC specializations in
`src/torchinferno/kernels/deepseek_v4_tilelang_definitions.py` adapt the
block layouts of vLLM's TileLang mHC kernels.

The DeepSeek V4 CUDA path in
`src/torchinferno/models/deepseek_v4/tensor_parallel.py` invokes SGLang's
precompiled RMSNorm operator when it is available.

The ordinary text-conversation framing in
`src/torchinferno/openai_server.py` follows vLLM's DeepSeek V4 encoding
contract for system, developer, user, and assistant messages.

Copyright contributors to the SGLang and vLLM projects

Licensed under the Apache License, Version 2.0. The complete license text is
included in `licenses/Apache-2.0.txt`.
