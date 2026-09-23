# profiling the matmul kernel

## ncu status on this machine

`ncu` is installed (cuda 12.6 toolkit in /usr/local/cuda) but profiling fails with
ERR_NVGPUCTRPERM: the wsl2 passthrough driver does not hand out performance counters
to this session. the windows side key RmProfilingAdminOnly is already 0, so the usual
remaining fix is `wsl --shutdown` to reinitialize the driver, which kills every
session including this one, so it has not been done yet.

the command to run once counters work:

```
ncu --set basic -k regex:matmul -c 3 ~/triton/triton-venv/bin/python prof_matmul.py
```

so everything below is arithmetic on numbers that were measured (cuda events and
nvidia-smi sampling), not hardware counters. the math is simple enough to check.

## what was measured

- kernel: BLOCK 64x64x32, num_stages=3, num_warps=4, 2048^3 fp16
- 0.67 - 0.71 ms across runs -> 24 - 25 TFLOPS (best config autotune picked)
- cublas on the same shape: 0.77 - 0.81 ms (21 - 22 TFLOPS)
- clocks sampled with nvidia-smi while the kernel loops: ~1980 MHz sustained
  (max 2100), ~57 W, 59 C
- device: RTX 3060 Laptop, 30 SMs, 3840 cuda cores, 3 MB L2

## roofline (computed)

- fp32 peak: 3840 * 2 * 1.98 GHz = 15.2 TFLOPS
- tensor fp16 peak on GA10x (dense, fp32 accumulate): 2x fp32 = 30.4 TFLOPS
- gemm flops: 2 * 2048^3 = 17.2 GFLOP -> ideal time 0.565 ms
- we measure ~0.7 ms -> about 80% of the tensor roofline

## memory side (computed)

- compulsory dram traffic is just A + B + C = 24 MB -> ~34 GB/s at our kernel time,
  nowhere near the ~336 GB/s the dram can do, so dram is not the bottleneck
- the naive re-read count is much worse: with 64x64 output tiles, A is re-read
  N/64 = 32 times and B too, ~520 MB total. that would need 740+ GB/s aggregate,
  impossible from dram alone, so L2 plus the order blocks happen to run in must be
  absorbing most of the re-reads. note the 3 MB L2 cannot even hold A or B (8 MB each)
- transpose.py showed this gpu does ~216 GB/s on a pure streaming pattern, which
  also rules out memory as the limiter here

## conclusion

the kernel is compute bound at roughly 80% of what the tensor cores can sustain at
the measured clock. the missing ~20% is most likely the smem -> register pipeline not
being fully saturated: BLOCK_K=32 means many short loop iterations, and 64x64 tiles
are on the small side. pushing tiles bigger runs into registers/smem on a 100 KB smem
part, which is probably why autotune settled on 64x64. confirming any of this needs
the real counters, hence the todo.

cublas being slower than my kernel at this exact shape is a heuristic/config thing,
it wins on other shapes.

## to finish this properly later

1. on windows: `wsl --shutdown` (reinit the driver, all wsl sessions die)
2. if counters are still denied, check RmProfilingAdminOnly is 0 in
   HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak (it already is here)
3. rerun the ncu command at the top and compare sm__pipe_tensor and dram__throughput
   with the computed numbers above
