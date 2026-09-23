# 练习：自己写个 relu，然后把 add 和 relu 融合到同一个 kernel 里
# 融合的好处是 x+y 的中间结果不用写回显存，少一次读写

import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


# add 和 relu 写在一起就是融合了，感觉就是把两段代码拼起来
@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x + y
    z = tl.where(z > 0, z, 0.0)
    tl.store(out_ptr + offs, z, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    relu_kernel[grid](x, out, n, BLOCK=BLOCK)
    return out


def triton_add_relu(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_relu_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8192, device="cuda")
    y = torch.randn(8192, device="cuda")

    out1 = triton_relu(x)
    print("relu 和 torch 一致吗:", torch.allclose(out1, torch.relu(x)))

    out2 = triton_add_relu(x, y)
    ref = torch.relu(x + y)
    print("add+relu 和 torch 一致吗:", torch.allclose(out2, ref))

    # TODO: 试试不同 BLOCK_SIZE 跑个时间对比
