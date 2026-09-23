# 第一个 triton 程序，向量加法，基本就是照着官方 tutorial 01 抄的
# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n  # 最后一个 block 可能超出 n，要用 mask 挡掉
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)  # 一共要几个 block
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 10000  # 故意不是 1024 的整数倍，测一下 mask
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    out = triton_add(x, y)
    print("和 torch 结果一致吗:", torch.allclose(out, x + y))
    print("最大误差:", (out - x - y).abs().max().item())
