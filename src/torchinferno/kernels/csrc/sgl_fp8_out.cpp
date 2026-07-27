#include <ATen/ATen.h>
#include <torch/library.h>

#include <optional>

// Exported by the version-pinned sglang-kernel SM90 provider. Its public
// Python wrapper allocates an output; this adapter exposes the same kernel
// with a caller-owned output so a symmetric-memory reduce buffer can be used.
void cutlass_scaled_mm_sm90_fp8(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scales,
    const at::Tensor& b_scales,
    const std::optional<at::Tensor>& bias);

at::Tensor fp8_scaled_mm_out(
    at::Tensor out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_scales,
    const at::Tensor& b_scales) {
  TORCH_CHECK(
      out.is_cuda() && a.is_cuda() && b.is_cuda() && a_scales.is_cuda() &&
          b_scales.is_cuda(),
      "FP8 scaled-mm tensors must be CUDA tensors");
  TORCH_CHECK(
      out.get_device() == a.get_device() && a.get_device() == b.get_device() &&
          a.get_device() == a_scales.get_device() &&
          a.get_device() == b_scales.get_device(),
      "FP8 scaled-mm tensors must be on one CUDA device");
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
      "FP8 scaled-mm matrices must be two-dimensional");
  TORCH_CHECK(
      a.size(0) >= 1 && a.size(0) <= 64,
      "FP8 scaled-mm output adapter supports 1 through 64 rows");
  TORCH_CHECK(a.size(1) == b.size(0), "FP8 scaled-mm inner dimensions must match");
  TORCH_CHECK(
      a.size(1) % 16 == 0 && b.size(1) % 16 == 0,
      "FP8 scaled-mm inner and output dimensions must be multiples of 16");
  TORCH_CHECK(
      out.size(0) == a.size(0) && out.size(1) == b.size(1),
      "FP8 scaled-mm output shape does not match the matrices");
  TORCH_CHECK(
      a.scalar_type() == at::kFloat8_e4m3fn &&
          b.scalar_type() == at::kFloat8_e4m3fn,
      "FP8 scaled-mm inputs must use float8_e4m3fn");
  TORCH_CHECK(
      a_scales.scalar_type() == at::kFloat &&
          b_scales.scalar_type() == at::kFloat,
      "FP8 scaled-mm scales must use float32");
  TORCH_CHECK(
      out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf,
      "FP8 scaled-mm output must use bfloat16 or float16");
  TORCH_CHECK(out.is_contiguous(), "FP8 scaled-mm output must be contiguous");
  TORCH_CHECK(a.is_contiguous(), "FP8 scaled-mm A matrix must be contiguous");
  TORCH_CHECK(
      b.stride(0) == 1 && b.stride(1) == b.size(0),
      "FP8 scaled-mm B matrix must be column-major contiguous");
  TORCH_CHECK(
      a_scales.dim() == 2 && a_scales.size(0) == a.size(0) &&
          a_scales.size(1) == 1 &&
          a_scales.is_contiguous(),
      "FP8 scaled-mm A scales must be contiguous [M, 1]");
  TORCH_CHECK(
      b_scales.dim() == 2 && b_scales.size(0) == 1 &&
          b_scales.size(1) == b.size(1) &&
          b_scales.is_contiguous(),
      "FP8 scaled-mm B scales must be contiguous [1, N]");
  cutlass_scaled_mm_sm90_fp8(out, a, b, a_scales, b_scales, std::nullopt);
  return out;
}

TORCH_LIBRARY(torchinferno_sgl_fp8, m) {
  m.def(
      "scaled_mm_out(Tensor(a!) out, Tensor a, Tensor b, Tensor a_scales, "
      "Tensor b_scales) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torchinferno_sgl_fp8, CUDA, m) {
  m.impl("scaled_mm_out", &fp8_scaled_mm_out);
}
