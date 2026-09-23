# 踩坑记录

- mask 忘写或写错会越界，出现过 nan。现在每次 load/store 之前都先想一下边界在哪。
- python 的 and / or 不能用在 tensor 上，要用 & 和 |，而且得加括号，不然优先级不对。
- softmax 要先减掉最大值再 exp，不然 fp32 都会溢出。
- tl.dot 的块最小是 16x16x16，BLOCK_K 选太小吞吐量上不去。
- 写 matmul 时假设了 K 是 BLOCK_K 的整数倍，K 方向边界没处理，以后有空补上。
- GPU 上计时要用 cuda event 加 synchronize，用 time.time() 量出来的是乱的（kernel 是异步的）。
- fp16 的 matmul 结果和 torch 差一点点是正常的，累加顺序不完全一样，比较时给容差。
- autotune 第一次调用会慢（它要把每个配置都跑一遍），之后再走缓存的 best_config。
