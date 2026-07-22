"""DeepSeek-V4 block-FP8, MXFP4, sparse-attention, and mHC kernels.

Adapted from ``deepseek-ai/DeepSeek-V4-Flash`` public inference/kernel.py at
commit 60d8d70770c6776ff598c94bb586a859a38244f1. The upstream implementation is
MIT licensed; see ``THIRD_PARTY_NOTICES.md``.
"""

import math

import torch
import tilelang
import tilelang.language as T
from typing import Optional


tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(pass_configs=pass_configs)
def q_norm_rope_kernel(
    heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float = 1e-6,
):
    """Fuse self-RMSNorm and interleaved RoPE for one-token Q decode."""
    tokens = T.symbolic("tokens")
    nope_dim = head_dim - rope_dim

    @T.prim_func
    def q_norm_rope_kernel_(
        q: T.Tensor[(tokens, heads, head_dim), BF16],
        freqs: T.Tensor[(rope_dim,), FP32],
        output: T.Tensor[(tokens, heads, head_dim), BF16],
    ):
        with T.Kernel(tokens * heads, threads=256) as row:
            token = row // heads
            head = row % heads
            values = T.alloc_fragment((head_dim,), FP32)
            T.copy(q[token, head, 0], values)

            square_sum = T.alloc_reducer((1,), FP32, replication="all")
            T.fill(square_sum, 0.0)
            for index in T.Parallel(head_dim):
                square_sum[0] += values[index] * values[index]
            T.finalize_reducer(square_sum)
            inverse_rms = T.rsqrt(square_sum[0] / head_dim + eps)

            for index in T.Parallel(nope_dim):
                output[token, head, index] = values[index] * inverse_rms

            thread = T.get_thread_binding()
            if thread < rope_dim // 2:
                index = nope_dim + thread * 2
                # Match the reference's in-place BF16 normalization before
                # its FP32 complex multiply.
                real = T.Cast(
                    FP32,
                    T.Cast(BF16, T.Cast(FP32, q[token, head, index]) * inverse_rms),
                )
                imag = T.Cast(
                    FP32,
                    T.Cast(
                        BF16,
                        T.Cast(FP32, q[token, head, index + 1]) * inverse_rms,
                    ),
                )
                cosine = freqs[thread * 2]
                sine = freqs[thread * 2 + 1]
                output[token, head, index] = real * cosine - imag * sine
                output[token, head, index + 1] = imag * cosine + real * sine

    return q_norm_rope_kernel_


@tilelang.jit(pass_configs=pass_configs)
def rope_inplace_kernel(
    heads: int,
    head_dim: int,
    rope_dim: int,
    inverse: bool = False,
):
    """Apply interleaved RoPE to the tail of a full-stride Q/K tensor."""
    tokens = T.symbolic("tokens")
    nope_dim = head_dim - rope_dim

    @T.prim_func
    def rope_inplace_kernel_(
        value: T.Tensor[(tokens, heads, head_dim), BF16],
        freqs: T.Tensor[(rope_dim,), FP32],
    ):
        with T.Kernel(tokens * heads, threads=32) as row:
            token = row // heads
            head = row % heads
            thread = T.get_thread_binding()
            index = nope_dim + thread * 2
            real = T.Cast(FP32, value[token, head, index])
            imag = T.Cast(FP32, value[token, head, index + 1])
            cosine = freqs[thread * 2]
            sine = (
                -freqs[thread * 2 + 1]
                if inverse
                else freqs[thread * 2 + 1]
            )
            value[token, head, index] = real * cosine - imag * sine
            value[token, head, index + 1] = imag * cosine + real * sine

    return rope_inplace_kernel_


@tilelang.jit(pass_configs=pass_configs)
def hc_post_kernel(
    hc: int,
    hidden: int,
    threads: int = 128,
    block: int = 1024,
):
    """Fuse mHC post-mapping into one BF16 output kernel.

    The block layout is adapted from vLLM's Apache-2.0 TileLang mHC kernel.
    """
    tokens = T.symbolic("tokens")
    block = math.gcd(hidden, block)

    @T.prim_func
    def hc_post_kernel_(
        value: T.Tensor[(tokens, hidden), BF16],
        residual: T.Tensor[(tokens, hc, hidden), BF16],
        post: T.Tensor[(tokens, hc), FP32],
        comb: T.Tensor[(tokens, hc, hc), FP32],
        output: T.Tensor[(tokens, hc, hidden), BF16],
    ):
        with T.Kernel(tokens, threads=threads) as token:
            residual_shared = T.alloc_shared((hc, block), BF16)
            value_shared = T.alloc_shared((block,), BF16)
            result = T.alloc_fragment((hc, block), FP32)
            residual_local = T.alloc_fragment((hc, block), FP32)
            value_local = T.alloc_fragment((block,), FP32)
            comb_local = T.alloc_fragment((hc, hc), FP32)
            post_local = T.alloc_fragment((hc,), FP32)
            T.copy(comb[token, 0, 0], comb_local)
            T.copy(post[token, 0], post_local)
            for block_index in T.Serial(T.ceildiv(hidden, block)):
                T.copy(
                    residual[token, 0, block_index * block],
                    residual_shared,
                )
                T.copy(value[token, block_index * block], value_shared)
                T.copy(residual_shared, residual_local)
                T.copy(value_shared, value_local)
                for channel, index in T.Parallel(hc, block):
                    result[channel, index] = (
                        post_local[channel] * value_local[index]
                    )
                    for source in T.vectorized(hc):
                        result[channel, index] += (
                            comb_local[source, channel]
                            * residual_local[source, index]
                        )
                T.copy(
                    result,
                    output[token, 0, block_index * block],
                )

    return hc_post_kernel_


@tilelang.jit(pass_configs=pass_configs)
def hc_prenorm_gemm_kernel(
    hidden: int,
    hc: int = 4,
    outputs: int = 24,
    threads: int = 1024,
    output_tile: int = 4,
):
    """Compute the mHC projection and residual squared norm together.

    Adapted from vLLM's Apache-2.0 TileLang mHC pre-norm GEMV.
    """
    tokens = T.symbolic("tokens")
    width = hc * hidden
    iterations = width // threads
    output_tiles = T.ceildiv(outputs, output_tile)

    @T.prim_func
    def hc_prenorm_gemm_kernel_(
        value: T.Tensor[(tokens, width), BF16],
        weight: T.Tensor[(outputs, width), FP32],
        output: T.Tensor[(tokens, outputs), FP32],
        square_sum: T.Tensor[(tokens,), FP32],
    ):
        with T.Kernel(tokens, output_tiles, threads=threads) as (token, tile):
            thread = T.get_thread_binding()
            accumulator = T.alloc_local((output_tile,), FP32)
            square = T.alloc_local((1,), FP32)
            T.clear(accumulator)
            T.clear(square)
            for iteration in T.serial(iterations):
                index = iteration * threads + thread
                input_value = value[token, index]
                for inner in T.unroll(output_tile):
                    output_index = tile * output_tile + inner
                    if output_index < outputs:
                        accumulator[inner] += (
                            input_value * weight[output_index, index]
                        )
                if tile == 0:
                    square[0] += input_value * input_value

            for inner in T.unroll(output_tile):
                accumulator[inner] = T.warp_reduce_sum(accumulator[inner])
            if tile == 0:
                square[0] = T.warp_reduce_sum(square[0])

            lane = thread % 32
            warp = thread // 32
            warps = threads // 32
            warp_output = T.alloc_shared((warps, output_tile), FP32)
            warp_square = T.alloc_shared((warps,), FP32)
            if lane == 0:
                for inner in T.unroll(output_tile):
                    warp_output[warp, inner] = accumulator[inner]
                if tile == 0:
                    warp_square[warp] = square[0]
            T.sync_threads()
            if warp == 0:
                if lane < output_tile:
                    reduced = T.alloc_var(FP32, init=0.0)
                    for source_warp in T.unroll(warps):
                        reduced += warp_output[source_warp, lane]
                    output_index = tile * output_tile + lane
                    if output_index < outputs:
                        output[token, output_index] = reduced
                if lane == 0 and tile == 0:
                    reduced_square = T.alloc_var(FP32, init=0.0)
                    for source_warp in T.unroll(warps):
                        reduced_square += warp_square[source_warp]
                    square_sum[token] = reduced_square

    return hc_prenorm_gemm_kernel_


def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32,
    round_scale=False, inplace=False
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16."""
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale or inplace else 2
    blk_m = 32
    group_size = block_size
    # Internal computation in FP32; scale_dtype controls output storage format.
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else out_dtype

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP8, T.clamp(
                                x_local[i, j] / s_local[i], fp8_min, fp8_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], fp8_min, fp8_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP)."""
    N = x.size(-1)
    assert N % block_size == 0
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)
    kernel = act_quant_kernel(
        N, block_size, scale_dtype=tl_dtype,
        round_scale=scale_fmt is not None, inplace=inplace,
    )
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp4_quant_kernel(
    N, block_size=32, in_dtype=BF16, scale_dtype=FE8M0, inplace=False
):
    """Block-wise FP4 quantization. Power-of-2 scale via bit ops. inplace=True does fused quant+dequant."""
    M = T.symbolic("M")
    fp4_max = 6.0
    fp4_max_inv = 1.0 / fp4_max
    blk_m = 32
    group_size = block_size
    compute_dtype = FP32
    out_dtype = in_dtype if inplace else FP4

    @T.prim_func
    def fp4_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), compute_dtype)
            s_local = T.alloc_fragment((blk_m,), compute_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=2):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
                if inplace:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.Cast(
                            out_dtype,
                            T.Cast(compute_dtype, T.Cast(FP4, T.clamp(
                                x_local[i, j] / s_local[i], -fp4_max, fp4_max
                            ))) * s_local[i],
                        )
                else:
                    for i, j in T.Parallel(blk_m, group_size):
                        y_local[i, j] = T.clamp(
                            x_local[i, j] / s_local[i], -fp4_max, fp4_max
                        )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = T.Cast(scale_dtype, s_local[i])
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return fp4_quant_kernel_


def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP4 quantization. inplace=True does fused quant+dequant back to BF16."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous()
    y = torch.empty_like(z) if inplace else z.new_empty(*z.shape[:-1], N // 2, dtype=torch.float4_e2m1fn_x2)
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=torch.float8_e8m0fnu)
    kernel = fp4_quant_kernel(N, block_size, inplace=inplace)
    kernel(z.view(-1, N), y.view(-1, y.size(-1)), s.view(-1, N // block_size))
    if inplace:
        x.copy_(y)
        return x
    return y, s


@tilelang.jit(pass_configs=pass_configs)
def fp8_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    assert out_dtype in [BF16, FP32]

    M = T.symbolic("M")
    group_size = 128
    block_M = 32
    block_N = 128
    block_K = 128

    @T.prim_func
    def fp8_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP8],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
        scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), FP32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Cast scales to FP32 for computation; scales_b has one value per block_N group
                Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Separate accumulator for scale-corrected results (2x accumulation precision)
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp8_gemm_kernel_


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-128 block FP8 scaling on both A and B."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
    kernel = fp8_gemm_kernel(N, K, scale_dtype=tl_dtype)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


@tilelang.jit(pass_configs=pass_configs)
def sparse_attn_kernel(h: int, d: int, scale=None):
    """Sparse multi-head attention via index gathering + online softmax (FlashAttention-style).
    For each (batch, seq_pos), gathers top-k KV positions by index, computes attention
    with numerically stable running max/sum, and includes a learnable attn_sink bias."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return sparse_attn_kernel_


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    b, s, h, d = q.size()
    # Pad heads to 16 for kernel efficiency (stripped after)
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base,
           pre.view(-1, hc_mult), post.view(-1, hc_mult), comb.view(-1, hc_mult, hc_mult))
    return pre, post, comb


@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """FP8 act x FP4 weight GEMM kernel.

    C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T

    Act: 1x128 quant on K (reduce dim), FP8 with configurable scale dtype
    Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale

    B is stored as [N, K//2] in float4_e2m1fn_x2, logical [N, K] in fp4.
    The FP4 values are packed along the K (last) dimension.

    Strategy: load FP4 sub-blocks of size [block_N, sub_K] (sub_K=32),
    cast FP4 to FP8 via float, then do FP8xFP8 GEMM.
    Apply act scale (per 128 on K) and weight scale (per 32 on K) to the accumulator.
    """
    M = T.symbolic("M")
    act_group_size = 128
    weight_group_size = 32
    block_M = 32
    block_N = 128
    block_K = 32   # matches weight_group_size for simple scale handling
    n_sub = act_group_size // block_K  # 4 sub-blocks per act scale group

    @T.prim_func
    def fp4_gemm_kernel_(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(N, K), FP4],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_dtype],
        scales_b: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_fp4_shared)
                # FP4->FP8 cast must go through FP32 to avoid ambiguous C++ overload
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                # Weight scale: per 32 on K, indexed by k (each k is one block_K=32)
                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[bx * block_N + i, k])

                # Act scale: per 128 on K, indexed by k // 4
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return fp4_gemm_kernel_


def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.
    A has per-128 act scale; B has per-32 E8M0 weight scale.
    B is stored as [N, K//2] in float4_e2m1fn_x2 (2 FP4 values per byte, packed along K)."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.bfloat16)
    kernel = fp4_gemm_kernel(N, K, scale_dtype=tl_dtype)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
