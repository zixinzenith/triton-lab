# global sum over a big array, the classic two-stage reduction:
# stage 1: each program sums one chunk, writes its partial sum
# stage 2: one program sums the partials
# (an atomic_add version exists too but float atomics are order-dependent,
# the two-stage one is deterministic, easier to check against torch)

import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def final_sum_kernel(partials_ptr, out_ptr, n_partials, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = tl.load(partials_ptr + offs, mask=offs < n_partials, other=0.0)
    tl.store(out_ptr, tl.sum(p, axis=0))


def triton_sum(x, BLOCK=1024):
    n = x.numel()
    n_partials = triton.cdiv(n, BLOCK)
    partials = torch.empty(n_partials, device=x.device, dtype=x.dtype)
    partial_sum_kernel[(n_partials,)](x, partials, n, BLOCK=BLOCK)

    BLOCK2 = triton.next_power_of_2(n_partials)
    out = torch.empty(1, device=x.device, dtype=x.dtype)
    final_sum_kernel[(1,)](partials, out, n_partials, BLOCK=BLOCK2)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    n = (1 << 24) + 123  # not a multiple of the block, mask handles the tail
    x = torch.randn(n, device="cuda")

    out = triton_sum(x)
    ref = x.sum()
    print("matches torch:", torch.isclose(out[0], ref, rtol=1e-4).item())
    print("rel err:", ((out[0] - ref) / ref).abs().item())

    # reduction is memory bound, both versions just read the array once
    from benchmark import bench
    t1 = bench(lambda: triton_sum(x))
    t2 = bench(lambda: x.sum())
    gb = n * 4 / 1e9
    print("sum %.1f MB: triton %.3f ms (%.0f GB/s), torch %.3f ms (%.0f GB/s)"
          % (gb * 1e3, t1, gb / (t1 / 1e3), t2, gb / (t2 / 1e3)))
