# first triton program, vector add, basically following the official tutorial
# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n  # the last block may go past n, the mask catches that
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x, y):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)  # how many blocks we need
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 10000  # on purpose not a multiple of 1024, to test the mask
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    out = triton_add(x, y)
    print("matches torch:", torch.allclose(out, x + y))
    print("max err:", (out - x - y).abs().max().item())
