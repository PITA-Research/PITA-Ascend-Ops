# FmaAddcmul：f32 精度算子与审计记录

## 范围

本文档仅记录 `Optry` 中的 direct ACLNN/AscendC 自定义算子；不包含 XLA pass、
thunk 或业务图接入。

算子接口：

```c
aclnnFmaAddcmulGetWorkspaceSize(self, tensor1, tensor2, value, out,
                                &workspace_size, &executor)
aclnnFmaAddcmul(workspace, workspace_size, executor, stream)
```

逐元素语义为：

```text
p   = RN_f32(value * tensor1)
out = RN_f32_fma(p, tensor2, self)
```

其中第一层缩放必须先完成 binary32 round-to-nearest-even；第二层必须与 host
`std::fmaf(p, tensor2, self)` 对齐。原生 `aclnnAddcmul` 不满足这项最终 FMA
rounding 语义，不能替代本算子。

## 实现与调用契约

- 支持 `ACL_FLOAT` tensor、`ACL_FLOAT` host scalar、rank 0 或任意静态 rank 的
  contiguous dense `ND` tensor。
- 三个输入和输出必须完全同 shape；不支持 broadcast、动态 shape、非连续 view、
  非 f32 dtype 或 output 与任一输入重叠。
- `aclnnFmaAddcmulGetWorkspaceSize` 在不满足契约时返回 `161002`，且不触碰 `out`。
  自定义 API wrapper 会读取 raw tensor address，并按 f32 byte range 检查 overlap。
- 正常路径 workspace 为 0；一个 ACLNN 执行调用对应一次 `fma_addcmul` AICore
  kernel launch。kernel 只用 local buffer，不创建 global temporary tensor。
- Ascend 910B vector MAC 在本路径不具备可验证的最终 FMA rounding；kernel 因此在
  单 kernel 内使用 product/two-sum residual 恢复 finite-normal 的最终 rounding。
  NaN/Inf/subnormal/overflow 分类走保护分支，避免 residual 展开产生无效中间值。

## 审计结果（2026-07-24）

审计环境：CANN 8.5.0、Ascend 910B、direct ACLNN API。构建和安装：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
bash ./build.sh
./build_out/custom_opp_openEuler_aarch64.run --quiet \
  --install-path=/data/CodeSpace/Optry/custom_opp_install
```

### 数值与契约

运行：

```bash
./tests/ut/run_fma_addcmul_precision.sh
```

测试使用 `aclrtMemcpy` 上传/回读 f32 exact bits；CPU reference 明确先做 volatile
f32 缩放，再调用 `std::fmaf`。非 NaN 输出逐 bit 比较，NaN 只比较分类。

| 审计项 | 结果 |
|---|---|
| capture scalar，n=1 | PASS，`0xb5441280` |
| capture batch，n=4096 | PASS，所有元素 `0xb5441280` |
| random cancellation | PASS，16,384 finite-normal 组 |
| scale random | PASS，`{2,1,-1,0.5,1.25}`，每组 16,384 元素 |
| ordinary random | PASS，16,384 元素 |
| signed zero、subnormal、NaN、Inf/overflow | PASS |
| finite-normal product overflow | PASS；此项在审计中发现并修复 |
| shape `[]`、`[1]`、`[2]`、`[4096]`、`[2,3,4]` | PASS |
| dtype（含 f16 scalar）、shape、broadcast、non-contiguous、output alias | PASS：明确错误且 out 不变 |
| workspace | PASS：所有有效调用均为 0 |

这说明已覆盖的输入可与 CPU reference 对齐；它不是全部 `2^32` f32 位模式的穷举
证明。NaN payload bit pattern 也不是接口契约。

### 性能

运行：

```bash
./tests/benchmark/run_fma_addcmul_benchmark.sh
```

测试固定为 4096 个 f32 元素、100 次 warmup、1000 次迭代，使用 device event 取
`device_avg_us`。同一环境的三次复测结果：

| 路径 | device 平均延迟 | 说明 |
|---|---:|---|
| `FmaAddcmul` | 310.38–310.58 us | 一次 custom kernel、workspace 0、正确 FMA 语义 |
| `aclnnAddcmul` | 5.70–6.69 us | 普通非 fused 语义，仅作速度参照 |

当前 custom kernel 约为原生参照的 46–54 倍（约 50 倍）。因此审计结论是：

- 数值语义、ABI、调用契约和零 workspace 均通过当前 direct 测试；
- **性能验收未通过**。单 kernel launch 不等于高性能；Dekker residual、多 local
  buffer 以及 kernel 内逐元素 special 分类使该实现不适合当前高频热路径；
- 当前仓库没有与 4-thunk 或 20-thunk 路径等价的 direct benchmark，不能把上述数字
  外推为相对该两条路径的收益。

后续优化应先减少 special classification 的逐元素标量循环、压缩临时 local buffer，并在
目标 ISA 上重新验证可用的最终 FMA intrinsic；每次变化都必须保留上述 bitwise 回归。

## 文件索引

```text
op_host/fma_addcmul.cpp                     OpDef、shape inference、tiling
op_host/fma_addcmul_tiling.h                tiling data
op_host/api/fma_addcmul_api.cpp             checked ACLNN wrapper
op_kernel/fma_addcmul.cpp                   AscendC 单 kernel 实现
tests/ut/fma_addcmul_precision_test.cc      精度和非法契约回归
tests/ut/run_fma_addcmul_precision.sh       UT 入口（临时产物在 /tmp）
tests/benchmark/fma_addcmul_benchmark.cc    4096 元素延迟测量
tests/benchmark/run_fma_addcmul_benchmark.sh benchmark 入口（临时产物在 /tmp）
```
