# TODO from the README: how much does software pipelining (num_stages) actually help matmul
# fixed BLOCK 128x64x32, only num_stages changes
# num_stages makes the compiler overlap the load of the next k tile with the current mma
# ran it: 1 -> 3 clearly helps, beyond that it gets noisy. laptop gpu, probably thermals,
# so single numbers are not trustworthy, look at the trend instead

import torch
import triton

import matmul as mm
from benchmark import bench


def run(a, b, c, ns):
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm.matmul_kernel[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                           num_stages=ns)


if __name__ == "__main__":
    torch.manual_seed(0)
    M = N = K = 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    flops = 2.0 * M * N * K

    for ns in [1, 2, 3, 4, 5]:
        t = bench(lambda: run(a, b, c, ns), times=20)
        print("num_stages=%d: %.3f ms (%.2f TFLOPS)" % (ns, t, flops / t / 1e9))

    t = bench(lambda: torch.matmul(a, b), times=20)
    print("cublas      : %.3f ms (%.2f TFLOPS)" % (t, flops / t / 1e9))

    run(a, b, c, 3)
    print("matches torch:", torch.allclose(c, torch.matmul(a, b), atol=1e-2, rtol=1e-2))
