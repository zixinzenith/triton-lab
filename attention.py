# flash attention 的核心思路：把 softmax 放进 kernel 里分块在线算，
# 不用把 M x N 的分数矩阵写回显存（这玩意儿是 O(N^2) 的显存开销）
# 这里是简化版：不带 causal mask，只支持自注意力的形状
# 参考: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale,
                M, N,
                stride_qm, stride_qd, stride_kn, stride_kd, stride_vn, stride_vd,
                stride_om, stride_od,
                D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < M

    q = tl.load(q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=m_mask[:, None], other=0.0)

    # 在线 softmax 的三个状态：每行当前的 max、sum(exp)、加权和
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(n_mask[None, :], s, -float("inf"))  # pad 的位置不参与 softmax

        # 关键就是这个 rescale：来了新的更大的 max，把之前的 sum 和 acc 缩一下
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(tl.float16), mask=m_mask[:, None])


def triton_attn(q, k, v, sm_scale, BLOCK_M=64, BLOCK_N=64):
    M, D = q.shape
    N = k.shape[0]
    o = torch.empty_like(q)
    grid = (triton.cdiv(M, BLOCK_M),)
    attn_kernel[grid](q, k, v, o, sm_scale, M, N,
                      q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                      v.stride(0), v.stride(1), o.stride(0), o.stride(1),
                      D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return o


def ref_attn(q, k, v, sm_scale):
    # fp32 参考答案
    s = torch.matmul(q.float(), k.float().T) * sm_scale
    return torch.softmax(s, dim=-1) @ v.float()


if __name__ == "__main__":
    torch.manual_seed(0)
    # N 故意不整除，测一下 mask
    M, N, D = 500, 700, 64
    sm_scale = 1.0 / D ** 0.5
    q = torch.randn((M, D), device="cuda", dtype=torch.float16)
    k = torch.randn((N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((N, D), device="cuda", dtype=torch.float16)

    out = triton_attn(q, k, v, sm_scale)
    ref = ref_attn(q, k, v, sm_scale)
    print("和 fp32 参考一致吗:", torch.allclose(out.float(), ref, atol=1e-2, rtol=1e-2))
    print("最大误差:", (out.float() - ref).abs().max().item())

    # 大一点的尺寸对比耗时，naive 版要先把 M x N 的分数矩阵算出来写显存
    M = N = 2048
    q = torch.randn((M, D), device="cuda", dtype=torch.float16)
    k = torch.randn((N, D), device="cuda", dtype=torch.float16)
    v = torch.randn((N, D), device="cuda", dtype=torch.float16)

    from benchmark import bench
    t1 = bench(lambda: triton_attn(q, k, v, sm_scale), times=20)
    def naive():
        s = torch.matmul(q, k.T) * sm_scale
        p = torch.softmax(s.float(), dim=-1).to(torch.float16)
        return p @ v
    t2 = bench(naive, times=20)
    t3 = bench(lambda: F.scaled_dot_product_attention(q, k, v), times=20)
    print("attention %dx%d: triton %.3f ms, torch naive %.3f ms, sdpa %.3f ms"
          % (M, N, t1, t2, t3))
