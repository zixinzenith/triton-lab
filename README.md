# triton-lab

preparing for gpu related jobs, learning kernel programming starting from triton. this repo is the practice ground.
most of the code follows the official tutorials with my own tweaks. every script prints a comparison against torch, benchmark.py does cuda event timing.

currently:

- vector_add.py vector add, the hello world
- relu.py my own relu, plus a fused add + relu version
- softmax.py row-wise softmax (naive, no inner blocking)
- matmul.py blocked matmul (tl.dot), later added an autotune version
- num_stages.py num_stages pipelining comparison on matmul
- attention.py simplified flash attention (online softmax, causal mask, multi-head)
- benchmark.py timing against the torch builtins
- notes.md gotchas i ran into

## environment

WSL2 + RTX 3060 Laptop, torch 2.14 / triton 3.8, both installed with pip.

## run

```
python vector_add.py
python relu.py
python softmax.py
python matmul.py
python num_stages.py
python attention.py
python benchmark.py
```

## todo

- [x] software pipelining on matmul (num_stages), how much does it recover
- [x] see how flash attention does the blocked softmax
- [x] causal mask and multi-head for attention.py
- [ ] learn ncu, profile matmul, find the bottleneck
- [ ] check what the official attention has that mine doesn't: GQA, dropout, better scheduling
