# triton-lab

准备找 GPU 方向的工作，从 Triton 入手学 kernel 编程，这里放练习代码。
大部分是照官方教程写的然后自己改了改，跑每个文件会打印和 torch 结果的对比，benchmark.py 里用 cuda event 计了时。

目前有：

- vector_add.py 向量加法，相当于 hello world
- relu.py 自己写的 relu，外加一个 add + relu 融合的版本
- softmax.py 按行的 softmax（朴素写法，没分块）
- matmul.py 分块矩阵乘法（tl.dot），后面加了个 autotune 版本
- num_stages.py matmul 上试 num_stages 流水线的对比
- attention.py flash attention 的简化版（online softmax，不带 causal mask）
- benchmark.py 和 torch 自带算子对比耗时
- notes.md 踩坑记录

## 环境

WSL2 + RTX 3060 Laptop，torch 2.14 / triton 3.8，都是 pip 装的。

## 运行

```
python vector_add.py
python relu.py
python softmax.py
python matmul.py
python num_stages.py
python attention.py
python benchmark.py
```

## 待办

- [x] matmul 开 software pipelining（num_stages），看看能追回多少差距
- [x] 看 flash attention 是怎么做分块 softmax 的
- [ ] 给 attention.py 加上 causal mask 和多头的支持
- [ ] 学一下 ncu，给 matmul 做个 profile，看瓶颈在哪个
