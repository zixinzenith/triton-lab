# 用 cuda event 简单对比一下自己写的 kernel 和 torch 自带的差多少
# 计时方法: 先 warmup 几次，然后把多次调用夹在两个 event 之间取平均

import torch
import triton

import vector_add as va
import relu as rl
import matmul as mm


def bench(fn, times=50):
    for _ in range(3):
        fn()  # warmup
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(times):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / times  # ms


def main():
    torch.manual_seed(0)
    BLOCK = 1024

    n = 1 << 24
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    t1 = bench(lambda: va.add_kernel[grid](x, y, out, n, BLOCK=BLOCK))
    t2 = bench(lambda: x + y)
    print("vector add  n=2^24: triton %.4f ms, torch %.4f ms" % (t1, t2))

    x2 = torch.randn(n, device="cuda")
    out2 = torch.empty_like(x2)
    t3 = bench(lambda: rl.relu_kernel[grid](x2, out2, n, BLOCK=BLOCK))
    t4 = bench(lambda: torch.relu(x2))
    print("relu        n=2^24: triton %.4f ms, torch %.4f ms" % (t3, t4))

    M = N = K = 2048
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    t5 = bench(lambda: mm.matmul_auto(a, b, c), times=20)
    t6 = bench(lambda: torch.matmul(a, b), times=20)
    flops = 2.0 * M * N * K
    # 本来以为会慢很多，结果这个尺寸下居然比 cublas 还快一点，
    # 可能是 cublas 对这个 shape 挑的 kernel 一般，换个大点的尺寸再看看
    print("matmul 2048^3 fp16: triton %.3f ms (%.2f TFLOPS), cublas %.3f ms (%.2f TFLOPS), triton/cublas = %.2f"
          % (t5, flops / t5 / 1e9, t6, flops / t6 / 1e9, t5 / t6))


if __name__ == "__main__":
    main()
