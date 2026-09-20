# PITA-Ascend-Ops

面向 PITA 可微物理引擎的 Ascend NPU 自定义算子库，基于 CANN / Ascend C 实现，提供 ACLNN 调用接口、精度测试与性能基准。

PITA-Ascend-Ops 聚焦 PITA 可微物理引擎中对 FP32 数值语义敏感的计算路径，当前覆盖逐行点积、补偿求和和逐元素融合乘加。项目通过显式控制乘加顺序与舍入行为，对齐各算子约定的 CPU 参考实现。

- **明确的数值语义**：分别定义规约顺序、补偿算法和 FMA 舍入契约。
- **批量计算支持**：Dot 算子支持逐行批量计算及共享向量输入。
- **可复现的验证**：提供直接调用 ACLNN 接口的精度测试和设备耗时基准。

[算子列表](#算子列表) · [环境要求](#环境要求) · [快速开始](#快速开始) · [测试与性能评估](#测试与性能评估) · [目录结构](#目录结构) · [文档索引](#文档索引)

## 算子列表

| 算子 | 数据类型 | 功能与数值语义 | 文档 |
| --- | --- | --- | --- |
| `DotFp32LeftOrderNoNeumaier` | FP32 | 逐行点积，按指定的 JAX CPU AArch64 参考实现控制乘加顺序 | [Dot 算子](doc/dot_fp32_left_order.md) |
| `DotFp32LeftOrderNeumaier` | FP32 | 逐行点积，使用 Neumaier 补偿求和降低相消误差 | [Dot 算子](doc/dot_fp32_left_order.md) |
| `FmaAddcmul` | FP32 | 逐元素计算 `fmaf(RN_f32(value * tensor1), tensor2, self)` | [FmaAddcmul 算子](doc/fma_addcmul.md) |

两个 Dot 算子支持 `[K] × [K]`、`[B..., K] × [K]`、`[K] × [B..., K]` 和 `[B..., K] × [B..., K]`。当两侧均带批量维度时，批量维度必须完全一致，不支持批量维度广播。NoNeumaier 的对齐目标是文档中指定的 CPU lowering 行为；Neumaier 使用不同的规约算法，不保证与 JAX CPU dot 逐位一致。

`FmaAddcmul` 要求输入和输出为同形状、连续布局的 FP32 张量，`value` 为 FP32 主机标量；不支持广播或输出与输入内存重叠。当前实现使用一次 kernel launch，workspace 为 0。**该算子的性能验收尚未通过**，具体测试条件、延迟及限制见[性能记录](doc/fma_addcmul.md#性能)。

## 环境要求

| 组件 | 要求或当前配置 |
| --- | --- |
| 硬件 | Ascend 910B，当前编译目标为 `ascend910b` |
| 操作系统 | Linux；现有 Dot 性能脚本包含 AArch64 库路径 |
| CANN | 安装匹配的开发工具及运行环境；当前构建预设路径为 `/usr/local/Ascend/cann-9.0.0` |
| 构建工具 | 建议 CMake 3.19 及以上，GNU Make、Python 3、支持 C++17 的 G++ |
| 运行环境 | 已安装配套驱动，测试时可访问 NPU 设备 |

构建参数位于 [CMakePresets.json](CMakePresets.json)：

| 参数 | 说明 |
| --- | --- |
| `ASCEND_CANN_PACKAGE_PATH` | CANN 安装路径，必须与实际加载的环境一致 |
| `ASCEND_COMPUTE_UNIT` | 编译目标，当前为 `ascend910b` |
| `vendor_name` | OPP 包的厂商标识，当前为 `optry` |

仓库名称为 `PITA-Ascend-Ops`，当前安装目录仍使用构建配置中的 `optry` 标识，因此下文环境脚本路径为 `vendors/optry/`。

现有 [FmaAddcmul 审计记录](doc/fma_addcmul.md#审计结果2026-07-24)基于 CANN 8.5.0；测试脚本未设置 `ASCEND_HOME_PATH` 时也默认使用 8.5.0。构建预设与历史验证环境不同，使用其他版本时需在目标环境重新执行精度和性能测试。

## 快速开始

进入本地 `PITA-Ascend-Ops` 仓库根目录后，在同一个 Bash 会话中依次执行以下步骤。

### 1. 配置环境

按实际安装位置修改 CANN 路径，并同步修改 `CMakePresets.json` 中的 `ASCEND_CANN_PACKAGE_PATH`：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### 2. 构建算子包

```bash
bash ./build.sh
```

构建产物位于 `build_out/`，安装包命名为 `custom_opp_<系统信息>.run`。`build.sh` 会清理已有的 `build_out/` 内容后重新构建。

### 3. 安装并加载算子

将算子包安装到仓库下的 `custom_opp_install/`，与现有测试脚本使用的路径保持一致：

```bash
# 默认构建配置生成一个 .run 安装包
opp_packages=("$PWD"/build_out/custom_opp_*.run)
bash "${opp_packages[0]}" --quiet --install-path="$PWD/custom_opp_install"
source "$PWD/custom_opp_install/vendors/optry/bin/set_env.bash"
```

新开终端后，需重新加载 CANN 和自定义算子的环境脚本。若调整安装目录或 `vendor_name`，需同步调整测试脚本中的路径。

### 4. 验证安装

```bash
bash tests/ut/run_precision.sh
bash tests/ut/run_fma_addcmul_precision.sh
```

测试脚本会编译并运行 ACLNN 测试程序。Dot 测试还依赖 `build_out/` 中生成的头文件和库，运行前请保留构建产物。

## 测试与性能评估

### 精度测试

| 测试入口 | 验证内容 |
| --- | --- |
| `tests/ut/run_precision.sh` | 两个 Dot 算子的 CPU 参考结果对齐、支持的形状及批量场景 |
| `tests/ut/run_fma_addcmul_precision.sh` | FMA 数值语义、特殊值、形状，以及非法 dtype、广播、非连续布局和内存重叠的拒绝行为 |

FmaAddcmul 测试对非 NaN 输出逐位比较，对 NaN 仅比较分类。详细覆盖范围见[测试说明](tests/README.md)和各算子文档。

### 性能基准

```bash
# Dot：对比自定义算子、原生 Mul + ReduceSum 和逐行 Dot
bash tests/benchmark/run_benchmark.sh

# FmaAddcmul：对比自定义算子与原生 aclnnAddcmul
bash tests/benchmark/run_fma_addcmul_benchmark.sh
```

Dot 基准结果写入 `tests/benchmark/results/dot_perf_results.csv`；FmaAddcmul 基准结果输出到终端。`device_avg_us` 为通过设备事件测得的平均耗时，单位为微秒。

原生算子作为性能参照时，其数值语义可能与自定义算子不同。例如，`aclnnAddcmul` 不满足 `FmaAddcmul` 要求的最终 FMA 舍入语义。比较结果时应同时记录硬件、CANN 版本、输入形状和数值契约。

## 目录结构

```text
PITA-Ascend-Ops/
├── op_host/              # Host 侧算子注册、形状推导与 tiling
│   └── api/              # 自定义 ACLNN 接口封装
├── op_kernel/            # Ascend C AICore kernel 实现
├── tests/
│   ├── ut/               # 精度与接口契约测试
│   └── benchmark/        # 性能基准与结果
├── doc/                  # 算子接口、数值语义与验证记录
├── cmake/                # CANN 自定义算子构建与打包配置
├── scripts/              # OPP 安装与升级脚本模板
├── CMakePresets.json     # 编译目标、CANN 路径与打包选项
└── build.sh              # 构建入口
```

## 文档索引

- [Dot 算子说明](doc/dot_fp32_left_order.md)：支持的形状、规约顺序、精度验证与性能基准。
- [FmaAddcmul 算子说明](doc/fma_addcmul.md)：ACLNN 接口、调用契约、精度审计与性能限制。
- [测试说明](tests/README.md)：测试程序、运行入口及基准对比项。
