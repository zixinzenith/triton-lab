# entry point for ncu, kept small so the profile only contains what we care about
# usage:
#   ncu --set basic -k regex:matmul -c 3 /usr/local/cuda/bin/python prof_matmul.py
#   (or whatever python has triton, here: ~/triton/triton-venv/bin/python)
# numbers below are for BLOCK 64x64x32, the config autotune picked in matmul.py

import torch
import triton

import matmul as mm

M = N = K = 2048
a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)
c = torch.empty((M, N), device="cuda", dtype=torch.float16)

grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
for _ in range(3):
    mm.matmul_kernel[grid](a, b, c, M, N, K,
                           a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                           c.stride(0), c.stride(1),
                           BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
torch.cuda.synchronize()
print("done")
