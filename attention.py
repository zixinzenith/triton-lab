# flash attention idea: do the softmax inside the kernel block by block (online),
# so the M x N score matrix never gets written to memory (that thing is O(N^2) memory)
# this version has causal mask and multi-head, shapes are [B, H, M, D]
# causal has a nice property: whole blocks above the diagonal are all -inf and can be
# skipped entirely, only the diagonal block needs the fine grained m >= n mask,
# which saves about half the compute
# reference: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale,
                B, H, M, N,
                stride_qb, stride_qh, stride_qm, stride_qd,
                stride_kb, stride_kh, stride_kn, stride_kd,
                stride_vb, stride_vh, stride_vn, stride_vd,
                stride_ob, stride_oh, stride_om, stride_od,
                CAUSAL: tl.constexpr, D: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_b = pid_bh // H
    offs_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < M

    q_base = q_ptr + offs_b * stride_qb + offs_h * stride_qh
    k_base = k_ptr + offs_b * stride_kb + offs_h * stride_kh
    v_base = v_ptr + offs_b * stride_vb + offs_h * stride_vh

    q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0)

    # online softmax state: running max, running sum of exp, weighted sum so far
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # causal: everything past the diagonal of this block is all -inf, skip it
    if CAUSAL:
        hi = tl.minimum(N, (pid_m + 1) * BLOCK_M)
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        # diagonal block: mask out future positions with m >= n,
        # padded positions must go too, or the result is quietly wrong (got bitten by this)
        if CAUSAL:
            s = tl.where((offs_m[:, None] >= offs_n[None, :]) & n_mask[None, :], s, -float("inf"))
        else:
            s = tl.where(n_mask[None, :], s, -float("inf"))

        # the rescale is the key trick: when a bigger max shows up,
        # shrink the old sum and acc by exp(m_old - m_new)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_b * stride_ob + offs_h * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(tl.float16), mask=m_mask[:, None])


def triton_attn(q, k, v, sm_scale, causal=False, BLOCK_M=64, BLOCK_N=64):
    # q,k,v: [B, H, M, D] / [B, H, N, D]
    B, H, M, D = q.shape
    N = k.shape[2]
    o = torch.empty_like(q)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    attn_kernel[grid](q, k, v, o, sm_scale, B, H, M, N,
                      *q.stride(), *k.stride(), *v.stride(), *o.stride(),
                      CAUSAL=causal, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return o


def ref_attn(q, k, v, sm_scale, causal=False):
    # fp32 reference
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v.float()


if __name__ == "__main__":
    torch.manual_seed(0)
    # M/N on purpose not divisible, to test the mask
    B, H, M, N, D = 2, 3, 500, 700, 64
    sm_scale = 1.0 / D ** 0.5
    q = torch.randn((B, H, M, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)

    for causal in [False, True]:
        out = triton_attn(q, k, v, sm_scale, causal=causal)
        ref = ref_attn(q, k, v, sm_scale, causal=causal)
        name = "causal" if causal else "plain "
        print(name, "matches fp32 ref:", torch.allclose(out.float(), ref, atol=1e-2, rtol=1e-2),
              " max err:", (out.float() - ref).abs().max().item())

    # bigger size for timing (causal)
    B, H, M, N, D = 1, 8, 2048, 2048, 64
    q = torch.randn((B, H, M, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, H, N, D), device="cuda", dtype=torch.float16)
    causal_mask = torch.triu(torch.ones(M, N, dtype=torch.bool, device="cuda"), 1)

    from benchmark import bench
    t1 = bench(lambda: triton_attn(q, k, v, sm_scale, causal=True), times=20)
    def naive():
        s = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
        p = torch.softmax(s.float().masked_fill(causal_mask, float("-inf")), dim=-1).to(torch.float16)
        return p @ v
    t2 = bench(naive, times=20)
    t3 = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), times=20)
    print("causal attention 1x8x2048x64: triton %.3f ms, torch naive %.3f ms, sdpa %.3f ms"
          % (t1, t2, t3))
