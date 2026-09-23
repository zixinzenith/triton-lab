# LayerNorm forward, one program per row
# follows the official tutorial 05 idea but the simple version: single block per row,
# mean / var / normalize / affine all in one pass, nothing leaves the registers
# N is padded up to BLOCK, and padded lanes need care (see the note on var below)

import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, N, eps,
                     stride_xm, stride_ym,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)

    # mean is fine with other=0.0 since the sum just misses the pad lanes
    mean = tl.sum(x, axis=0) / N
    # but for var the pad lanes hold x - mean, which is NOT zero, must zero them out
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask)
    b = tl.load(b_ptr + cols, mask=mask)
    y = (x - mean) * rstd * w + b

    tl.store(y_ptr + row * stride_ym + cols, y, mask=mask)


def triton_layer_norm(x, w, b, eps=1e-5):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    layernorm_kernel[(M,)](x, y, w, b, N, eps, x.stride(0), y.stride(0), BLOCK=BLOCK)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 500  # 500 cols, not a power of two, on purpose
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    out = triton_layer_norm(x, w, b)
    ref = torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5)
    print("matches torch:", torch.allclose(out, ref, atol=1e-4))
    print("max err:", (out - ref).abs().max().item())

    # bigger size for timing, this op is memory bound
    M, N = 8192, 768
    x = torch.randn(M, N, device="cuda")
    w = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    from benchmark import bench
    t1 = bench(lambda: triton_layer_norm(x, w, b))
    t2 = bench(lambda: torch.nn.functional.layer_norm(x, (N,), w, b, 1e-5))
    print("layernorm 8192x768: triton %.3f ms, torch %.3f ms" % (t1, t2))
