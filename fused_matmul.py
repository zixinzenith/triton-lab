# matmul with a bias + activation epilogue baked into the kernel
# started with relu only, then gelu turned out to be 95% the same code,
# so the activation became a constexpr flag: triton compiles a separate kernel
# for each value, no runtime branch cost
# this is the thing triton is actually great at: pytorch runs relu(x @ w + b) as
# three kernels and round-trips the M x N intermediate through memory twice,
# here the epilogue rides along in registers for free

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_bias_act_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
                           stride_am, stride_ak, stride_bk, stride_bn,
                           stride_cm, stride_cn,
                           ACT: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    acc = acc + bias[None, :]

    if ACT == 0:  # relu
        acc = tl.where(acc > 0, acc, 0.0)
    else:  # gelu, tanh approximation (same formula as F.gelu(approximate='tanh'))
        # wrote tanh by hand: exp overflows to inf for big inputs and the formula
        # still lands on +-1, no nan
        inner = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
        e = tl.exp(2.0 * inner)
        tanh = 1.0 - 2.0 / (e + 1.0)
        acc = 0.5 * acc * (1.0 + tanh)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def fused_matmul(a, w, bias, act="relu", BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    K, N = w.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_bias_act_kernel[grid](a, w, bias, c, M, N, K,
                                 a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                                 c.stride(0), c.stride(1),
                                 ACT=0 if act == "relu" else 1,
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    # odd sizes to test the mask, K stays a multiple of BLOCK_K
    M, N, K = 500, 600, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    w = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)

    out = fused_matmul(a, w, bias, "relu")
    ref = torch.relu(a @ w + bias)
    print("relu matches torch:", torch.allclose(out, ref, atol=1e-2, rtol=1e-2))

    out = fused_matmul(a, w, bias, "gelu")
    ref = F.gelu(a @ w + bias, approximate="tanh")
    print("gelu matches torch:", torch.allclose(out, ref, atol=1e-2, rtol=1e-2))
    print("gelu max err:", (out - ref).abs().max().item())

    # timing: big square (gemm dominates) vs skinny (intermediate round trips hurt)
    M = N = K = 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    w = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)

    from benchmark import bench
    t1 = bench(lambda: fused_matmul(a, w, bias, "relu", 128, 64, 32), times=20)
    t2 = bench(lambda: torch.relu(a @ w + bias), times=20)
    print("bias+relu 2048^3: fused %.3f ms, torch %.3f ms" % (t1, t2))

    M, N, K = 16384, 256, 512
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    w = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    t3 = bench(lambda: fused_matmul(a, w, bias, "gelu", 64, 64, 32), times=20)
    t4 = bench(lambda: F.gelu(a @ w + bias, approximate="tanh"), times=20)
    print("bias+gelu 16384x256x512: fused %.3f ms, torch %.3f ms" % (t3, t4))
