# transpose looks trivial but it is THE lesson on memory coalescing:
# reading rows and writing columns means one side of the access is strided,
# and strided access wastes almost all of each 128B memory transaction
# version 1 (row): each program moves one whole row, the store is fully scattered
# version 2 (tiled): move BLOCK x BLOCK tiles, transpose in registers with tl.trans,
# so both the load and the store are coalesced

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_row(x_ptr, y_ptr, N,
                  stride_xm, stride_xn, stride_yn, stride_ym,
                  BLOCK: tl.constexpr):
    # one program per row of x (one column of y)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask)
    tl.store(y_ptr + offs * stride_yn + row * stride_ym, x, mask=mask)


@triton.jit
def transpose_tiled(x_ptr, y_ptr, M, N,
                    stride_xm, stride_xn, stride_ym, stride_yn,
                    BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
                mask=mask, other=0.0)
    xt = tl.trans(x)  # transpose in registers, keeps both sides coalesced
    tl.store(y_ptr + offs_n[:, None] * stride_ym + offs_m[None, :] * stride_yn,
             xt, mask=(offs_n[:, None] < N) & (offs_m[None, :] < M))


def triton_transpose_row(x):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    BLOCK = triton.next_power_of_2(N)
    transpose_row[(M,)](x, y, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                        BLOCK=BLOCK)
    return y


def triton_transpose_tiled(x, BLOCK=32):
    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    transpose_tiled[grid](x, y, M, N, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
                          BLOCK=BLOCK)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    # odd sizes to test the mask
    x = torch.randn(500, 700, device="cuda")
    ref = x.t().contiguous()
    print("row   matches torch:", torch.equal(triton_transpose_row(x), ref))
    print("tiled matches torch:", torch.equal(triton_transpose_tiled(x), ref))

    # bigger size for timing, fp32, this op is pure memory traffic
    x = torch.randn(4096, 4096, device="cuda")
    ref = x.t().contiguous()

    from benchmark import bench
    t1 = bench(lambda: triton_transpose_row(x), times=20)
    t2 = bench(lambda: triton_transpose_tiled(x), times=20)
    t3 = bench(lambda: x.t().contiguous(), times=20)
    # each version moves 2 * M * N * 4 bytes (read + write)
    bw = lambda t: 2 * 4096 * 4096 * 4 / (t / 1e3) / 1e9
    print("transpose 4096^2: row %.3f ms (%.0f GB/s), tiled %.3f ms (%.0f GB/s), torch %.3f ms (%.0f GB/s)"
          % (t1, bw(t1), t2, bw(t2), t3, bw(t3)))
