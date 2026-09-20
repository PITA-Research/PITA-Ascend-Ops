# 测试说明

这个目录包含当前维护中的精度单测和性能 benchmark，按用途分成两个子目录：

```text
ut/          精度和语义单测
benchmark/   性能 benchmark
```

## 精度测试

`ut/precision_aclnn.cc` 覆盖两个 ACLNN 自定义算子，并验证支持的 shape 形式：

```text
[K] x [K]
[B..., K] x [K]
[K] x [B..., K]
[B..., K] x [B..., K]
```

验证目标：

```text
DotFp32LeftOrderNoNeumaier == JAX CPU AArch64 fp32 dot reference
DotFp32LeftOrderNeumaier   == CPU fp32 Neumaier reference
```

运行：

```bash
cd /data/CodeSpace/Optry
./tests/ut/run_precision.sh
```

`ut/fma_addcmul_precision_test.cc` 使用 direct ACLNN API 验证 FMA 逐 bit 语义：真实
capture、随机相消、缩放、特殊值、五种 shape，以及 dtype/shape/broadcast/非连续/alias
拒绝（失败时不写 output）。运行：

```bash
cd /data/CodeSpace/Optry
./tests/ut/run_fma_addcmul_precision.sh
```

FmaAddcmul 的单算子延迟（4096 元素）与原生 `aclnnAddcmul` 参照：

```bash
./tests/benchmark/run_fma_addcmul_benchmark.sh
```

## 性能 benchmark

`benchmark/benchmark_aclnn.cc` 对比：

```text
custom_no_neumaier      单次 batched DotFp32LeftOrderNoNeumaier ACLNN 调用
custom_neumaier         单次 batched DotFp32LeftOrderNeumaier ACLNN 调用
native_mul_reduce_sum   Ascend aclnnMul + aclnnReduceSum batched baseline
native_dot_loop         Ascend aclnnDot 按 batch row 循环调用
```

运行：

```bash
cd /data/CodeSpace/Optry
./tests/benchmark/run_benchmark.sh
```

结果输出到：

```text
tests/benchmark/results/dot_perf_results.csv
```
