#!/bin/bash
# Rebuild torchao from github source against our CUSTOM PyTorch (2.13.0a0, CUDA
# 12.6), producing cutlass kernels (_C_cutlass_90a.abi3.so) that actually LOAD --
# the prebuilt torchao 0.17 wheel was built for CUDA 13 / a stable torch ABI and
# its cutlass .so fails to load against our torch (the long-standing "torchao
# ABI-blocked" note in docs/PERF_GAP_ANALYSIS.md). This recipe resolves that.
#
# Result (torchao 0.18.0+git5165bfb): wheel includes a 10MB _C_cutlass_90a.abi3.so
# that loads cleanly (ctypes.CDLL OK); torch.ops.torchao cutlass ops are live
# (e.g. rowwise_scaled_linear_sparse_cutlass_f8f8). NOTE: this version's cutlass
# kernels are FP8-SPARSE (f8f8); it does NOT ship an int4 batched-marlin GEMM
# (no marlin_qqq_gemm), so the int4 DECODE lever still needs a different kernel.
# The pip index reachable here only has ancient torchao (0.0.1-0.1), so the
# github source build is the way to get a current, ABI-matching torchao.
#
#   bash scripts/build_torchao.sh
#   python -m pip install /tmp/ao_build/torchao-*.whl --no-deps --force-reinstall
set -e
export HTTPS_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080
export no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/usr/local/cuda-12.6
export MAX_JOBS=32
rm -rf /tmp/ao_build && mkdir -p /tmp/ao_build && cd /tmp/ao_build
git clone --depth 1 https://github.com/pytorch/ao.git
cd ao
git submodule update --init --recursive third_party/cutlass
python -m pip wheel --no-build-isolation --no-deps . -w /tmp/ao_build
ls -la /tmp/ao_build/*.whl
