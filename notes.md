# 踩坑记录

- mask 忘写或写错会越界，出现过 nan。现在每次 load/store 之前都先想一下边界在哪。
- python 的 and / or 不能用在 tensor 上，要用 & 和 |，而且得加括号，不然优先级不对。
- softmax 要先减掉最大值再 exp，不然 fp32 都会溢出。
- tl.dot 的块最小是 16x16x16，BLOCK_K 选太小吞吐量上不去。
- 写 matmul 时假设了 K 是 BLOCK_K 的整数倍，K 方向边界没处理，以后有空补上。
- GPU 上计时要用 cuda event 加 synchronize，用 time.time() 量出来的是乱的（kernel 是异步的）。
- fp16 的 matmul 结果和 torch 差一点点是正常的，累加顺序不完全一样，比较时给容差。
- autotune 第一次调用会慢（它要把每个配置都跑一遍），之后再走缓存的 best_config。
- flash attention 的精髓是 online softmax：kernel 里保存每行的 max 和 sum(exp)，每算完一块就 rescale（alpha = exp(m_old - m_new)），这样不用把 M x N 的分数矩阵写回显存。
- torch.allclose 不能直接比 fp16 和 fp32 的 tensor，要先 .float() 转过来（今天踩的）。
- 跑分时单次数字不可信，笔记本卡温度一高就降频，多跑几遍看趋势。
- causal attention 的好处不光是省一半计算：上三角的块连 k/v 都不用 load，整块跳过；只有对角线那块需要 m >= n 的细粒度 mask。注意 pad 的位置也要一起挡掉，不然算出来是错的（踩过）。
- 加了多头和 causal 之后和 sdpa 比还是差一点（0.335 vs 0.290 ms），官方 kernel 的调度更精细，暂时能接受。
