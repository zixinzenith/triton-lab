# gotchas

- forgetting the mask or writing it wrong causes out of bounds reads and nan. now i think about the boundaries before every load/store.
- python and/or do not work on tensors, need & and |, and parentheses matter (precedence).
- softmax must subtract the row max before exp, otherwise it overflows even in fp32.
- tl.dot minimum tile is 16x16x16, too small BLOCK_K kills throughput.
- my matmul assumes K is a multiple of BLOCK_K, no boundary handling on K yet, fix someday.
- timing on gpu needs cuda events + synchronize, time.time() gives garbage (kernels are async).
- fp16 matmul differs from torch by a tiny bit, different accumulation order, compare with a tolerance.
- autotune is slow on the first call (it benchmarks every config), after that it reuses the cached best_config.
- single benchmark numbers are not trustworthy, the laptop gpu throttles when it heats up, run a few times and look at the trend.
- flash attention is all about online softmax: keep a running max and a running sum of exp per row, rescale after each block (alpha = exp(m_old - m_new)), then the M x N score matrix never hits memory.
- torch.allclose refuses to compare fp16 vs fp32 tensors, .float() first (stepped on this today).
- causal attention saves more than half the compute: whole blocks above the diagonal get skipped without even loading k/v, only the diagonal block needs the fine grained m >= n mask. padded positions must be masked together with the causal condition, otherwise the result is silently wrong (stepped on this one too).
- after multi-head + causal i am still a bit behind sdpa (0.335 vs 0.290 ms), their kernel scheduling is more refined, acceptable for now.
