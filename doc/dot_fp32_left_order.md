# DotFp32LeftOrder 算子

本文档记录当前仓库中第一批 MJX Ascend 自定义算子：两个 fp32 dot 算子。

## 算子列表

```text
DotFp32LeftOrderNoNeumaier(lhs, rhs) -> out
DotFp32LeftOrderNeumaier(lhs, rhs) -> out
```

生成的 ACLNN 接口：

```text
aclnnDotFp32LeftOrderNoNeumaier
aclnnDotFp32LeftOrderNeumaier
```

## 支持的 shape

两个算子都支持以下 row-wise dot 形式：

```text
[K] x [K]                         -> scalar
[B..., K] x [K]                   -> [B...]
[K] x [B..., K]                   -> [B...]
[B..., K] x [B..., K]             -> [B...]
```

约束：

- 两个输入最后一维 `K` 必须相同。
- 如果两边都是 batched，即 `[B..., K] x [B..., K]`，leading dims 必须完全一致。
- 如果一边是 `[K]`，该输入会作为共享向量复用于另一边所有 batch row。
- 当前不做 batch broadcasting。

## 数值语义

### DotFp32LeftOrderNoNeumaier

NoNeumaier 用于对齐 MJX dump 中观察到的 JAX CPU AArch64 fp32 dot 参考结果。

对 `K=18` 的 MJX 关键路径，JAX CPU AArch64 lowering 的实际行为是：

```text
k = 0       单独 fp32 product
k = 1..7    分离的 fp32 multiply + fp32 add
k = 8..17   fused multiply-add
```

因此当前 NoNeumaier kernel 不是纯左到右 `sum += a*b`，而是按上述 CPU lowering 语义实现，以获得 bitwise 对齐。

### DotFp32LeftOrderNeumaier

Neumaier 版本使用相同的 row-wise dot 输入输出语义，但规约算法是 fp32 Neumaier compensated sum：

```text
product = lhs[k] * rhs[k]
next_sum = sum + product
correction = ...
compensation += correction
sum = next_sum
out = sum + compensation
```

它的目标是降低 dot 后的 fp32 取消误差。由于补偿算法改变了规约语义，它不保证等于 NoNeumaier 或 JAX CPU dot 的 bitwise 结果。

## 精度验证

精度单测位于：

```text
tests/ut/precision_aclnn.cc
```

运行：

```bash
cd /data/CodeSpace/Optry
./tests/ut/run_precision.sh
```

验证内容：

```text
DotFp32LeftOrderNoNeumaier == JAX CPU AArch64 fp32 dot reference
DotFp32LeftOrderNeumaier   == CPU fp32 Neumaier reference
```

测试覆盖：

```text
[K] x [K]
[B..., K] x [K]
[K] x [B..., K]
[B..., K] x [B..., K]
batch=1024, K=18
batch=4096, K=18
```

测试同时打印相对 float64 dot 的最大误差，用于评估 Neumaier 补偿效果。

## 性能 benchmark

benchmark 位于：

```text
tests/benchmark/benchmark_aclnn.cc
```

运行：

```bash
cd /data/CodeSpace/Optry
./tests/benchmark/run_benchmark.sh
```

对比项：

```text
custom_no_neumaier      单次 batched DotFp32LeftOrderNoNeumaier ACLNN 调用
custom_neumaier         单次 batched DotFp32LeftOrderNeumaier ACLNN 调用
native_mul_reduce_sum   Ascend aclnnMul + aclnnReduceSum batched baseline
native_dot_loop         Ascend aclnnDot 按 batch row 循环调用
```

结果输出：

```text
tests/benchmark/results/dot_perf_results.csv
```

`device_avg_us` 表示通过 device event 统计的单次对应 shape 计算平均耗时，单位是微秒。
