# 按行做 softmax，参考了官方 tutorial 02，但是用的是最朴素的写法（没有分块）
# 每一行丢给一个 program，列数不够 2 的幂的地方用 mask 补上

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)

    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < n_cols
    in_ptrs = in_ptr + row * in_row_stride + col_offs
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    # 先减最大值，不然 exp 会溢出
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    e = e / tl.sum(e, axis=0)

    out_ptrs = out_ptr + row * out_row_stride + col_offs
    tl.store(out_ptrs, e, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    # 故意用 500 列，不是 2 的幂，看看 mask 对不对
    x = torch.randn(16, 500, device="cuda")

    out = softmax(x)
    ref = torch.softmax(x, dim=1)
    print("和 torch 一致吗:", torch.allclose(out, ref))
    print("最大误差:", (out - ref).abs().max().item())
    print("每行加起来等于 1 吗:", out.sum(dim=1))
