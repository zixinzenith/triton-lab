# triton-lab

记录一下学 Triton 的过程，代码大部分是照着官方教程写的，然后自己改了改。跑每个文件会打印和 torch 结果的对比。

目前写了：

- vector_add.py 向量加法，相当于 hello world
- relu.py 自己写的 relu，外加一个 add + relu 融合的版本
- softmax.py 按行的 softmax（朴素写法，没分块）

## 环境

WSL2 + RTX 3060 Laptop，torch 和 triton 都是 pip 装的。

## 运行

```
python vector_add.py
python relu.py
python softmax.py
```

## 待办

- [ ] 换不同的 BLOCK_SIZE 对比一下耗时
- [ ] 看 tutorial 03 的 matmul，那个好像要懂一点 tiling
