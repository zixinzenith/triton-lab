# README 里那条待办：matmul 开 software pipelining (num_stages) 到底能快多少
# 固定 BLOCK 128x64x32 不动，只改 num_stages 对比
# num_stages 就是让下一块 k 的 load 和当前块的乘加重叠起来（编译器做流水线）
# 跑下来 1→3 有提升，再往上就不稳定了，笔记本卡估计是温度/降频的影响，
# 跑分最好多跑几遍看趋势，单次的数字不太可信

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
    print("结果和 torch 一致吗:", torch.allclose(c, torch.matmul(a, b), atol=1e-2, rtol=1e-2))
