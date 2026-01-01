# GPU Performance Optimization Cookbook

一个全面的 GPU 性能分析和 CUDA 优化实践指南，涵盖从基础到高级的 GPU 编程技巧和性能分析方法。

## 📚 项目概述

本项目是一个系统性的 GPU 性能分析和优化学习资源，包含三个核心模块：

1. **GPU 性能分析** (`gpu_profile/`) - 使用 Nsight Systems 进行性能分析
2. **GPU 优化实践** (`optimize_in_gpu/`) - 常见算子的优化技巧和实现
3. **CUDA 最佳实践** (`universe_best_cuda_practice/`) - 从基础到高级的 CUDA 编程实践

## 🗂️ 项目结构

```
nsight-systems-cookbook/
├── gpu_profile/                    # GPU 性能分析工具和示例
│   ├── example*.py                 # 各种性能分析示例脚本
│   ├── profile_example*.sh         # 性能分析脚本
│   ├── QUICK_START.md              # 快速入门指南
│   ├── GPU_PROFILING_GUIDE.md      # 性能分析详细指南
│   └── README_nsys.md              # Nsight Systems 使用说明
│
├── optimize_in_gpu/                # GPU 算子优化实践
│   ├── reduce/                     # Reduce 算子优化（7个版本）
│   ├── sgemm/                      # 矩阵乘法优化
│   ├── sgemv/                      # 矩阵向量乘法优化
│   ├── elementwise/                # 逐元素操作优化
│   ├── spmm/                       # 稀疏矩阵乘法
│   └── spmv/                       # 稀疏矩阵向量乘法
│
└── universe_best_cuda_practice/    # CUDA 最佳实践
    ├── 1_cuda_reduce_study/        # Reduce 优化研究
    ├── 2_cuda_sgemm_study/         # SGEMM 优化研究
    ├── 3_kernel_profiling_guide/   # Kernel 性能分析指南
    ├── 4_tensor_core_wmma/         # Tensor Core WMMA 使用
    ├── 5_mma_and_swizzle/          # MMA 指令和 Swizzle 优化
    ├── 6_cutlass_study/            # CUTLASS 库学习
    └── flash_attention/            # Flash Attention 实现
```

## 🚀 快速开始

### 1. GPU 性能分析

如果你想学习如何使用 Nsight Systems 进行 GPU 性能分析：

```bash
cd gpu_profile
# 查看快速入门指南
cat QUICK_START.md

# 运行示例
python example1_memory_allocation.py

# 进行性能分析
nsys profile --trace=cuda,nvtx --output=profile.nsys-rep python example1_memory_allocation.py

# 查看结果
nsys-ui profile.nsys-rep
```

**推荐阅读顺序：**
1. `QUICK_START.md` - 5分钟快速入门
2. `README_nsys.md` - Nsight Systems 详细使用说明
3. `GPU_PROFILING_GUIDE.md` - 完整的性能分析指南

### 2. GPU 算子优化

如果你想学习如何优化 GPU 算子：

```bash
cd optimize_in_gpu

# Reduce 优化示例（最详细的优化教程）
cd reduce
make
./bin/reduce_v0  # baseline 版本
./bin/reduce_v7  # 最终优化版本

# 查看详细的优化说明
cat README.md
```

**主要优化算子：**
- **Reduce** - 7个优化版本，从 baseline 到 shuffle 指令优化，带宽利用率达到 95.3%
- **SGEMM** - 矩阵乘法优化，性能达到 cuBLAS 的 96.8%
- **SGEMV** - 矩阵向量乘法优化，针对不同数据形状的优化策略
- **Elementwise** - 向量化内存访问优化

### 3. CUDA 最佳实践

如果你想系统学习 CUDA 编程：

```bash
cd universe_best_cuda_practice

# 使用 CMake 编译
mkdir build && cd build
cmake ..
make

# 运行示例
./1_cuda_reduce_study/my_reduce_v0_global_memory
```

**学习路径：**
1. `1_cuda_reduce_study/` - Reduce 算子从基础到高级优化
2. `2_cuda_sgemm_study/` - SGEMM 的 8 个优化版本
3. `3_kernel_profiling_guide/` - Kernel 性能分析和 Roofline 模型
4. `4_tensor_core_wmma/` - Tensor Core 使用
5. `5_mma_and_swizzle/` - 高级优化技巧
6. `6_cutlass_study/` - CUTLASS 库实践

## 📖 核心内容详解

### GPU 性能分析 (`gpu_profile/`)

这个模块提供了完整的 GPU 性能分析工具链：

- **7 个性能分析示例**：涵盖内存分配、数据传输、同步、内核开销、负载均衡、内存访问等常见问题
- **Nsight Systems 使用指南**：从基础到高级的使用方法
- **SQLite 查询工具**：用于程序化分析性能数据
- **NVTX 标记示例**：学习如何在代码中添加性能标记

**关键文件：**
- `QUICK_START.md` - 快速入门（5分钟）
- `GPU_PROFILING_GUIDE.md` - 完整指南
- `README_nsys.md` - Nsight Systems 参数详解
- `SQLITE_QUERY_GUIDE.md` - SQLite 数据查询方法

### GPU 算子优化 (`optimize_in_gpu/`)

这个模块展示了如何系统性地优化 GPU 算子，以 **Reduce** 算子为例：

#### Reduce 优化历程（7个版本）

| 版本 | 优化技巧 | 带宽 (GB/s) | 加速比 | 说明 |
|------|---------|------------|--------|------|
| v0 | Baseline | 159.2 | 1.0x | 基础实现，存在 warp divergence |
| v1 | 消除 warp divergence | 252.8 | 1.59x | 解决分支发散问题 |
| v2 | 消除 bank 冲突 | 321.7 | 1.27x | 优化 shared memory 访问 |
| v3 | 利用 idle 线程 | 600.4 | 1.86x | 增加每个线程的工作量 |
| v4 | 展开最后一维 | 756.4 | 1.26x | 减少同步开销 |
| v5 | 完全展开循环 | 767.6 | 1.01x | 减少循环开销 |
| v6 | 优化 block 数量 | 768.1 | 1.00x | 合理设置 block 数量 |
| v7 | Shuffle 指令 | 770.3 | 1.00x | 使用 warp shuffle 指令 |

**最终性能：** 带宽利用率达到 **95.3%** (858GB/s / 900GB/s)

详细优化过程请参考：`optimize_in_gpu/reduce/README.md`

#### 其他算子优化

- **SGEMM**: 达到 cuBLAS 96.8% 的性能，包含 CUDA C 级别和 SASS 级别的优化
- **SGEMV**: 针对不同数据形状（n=32, n<32, n>32）的优化策略
- **Elementwise**: 向量化内存访问（float, float2, float4）的性能对比

### CUDA 最佳实践 (`universe_best_cuda_practice/`)

这个模块提供了系统性的 CUDA 学习路径：

1. **Reduce 研究** - 10 个版本的 reduce 实现，从全局内存到 shuffle 指令
2. **SGEMM 研究** - 8 个版本的矩阵乘法优化，包括：
   - 全局内存版本
   - 共享内存版本
   - 滑动窗口优化
   - Float4 向量化
   - 寄存器外积
   - 双缓冲技术
3. **Kernel 性能分析** - Transpose 算子的优化和 Roofline 模型分析
4. **Tensor Core** - WMMA API 的使用和优化
5. **MMA 和 Swizzle** - 高级内存访问优化
6. **CUTLASS** - NVIDIA 高性能 GEMM 库的使用
7. **Flash Attention** - 注意力机制的高效实现

## 🎯 学习路径建议

### 初学者路径

1. **第一步：了解 GPU 性能分析**
   ```bash
   cd gpu_profile
   # 阅读 QUICK_START.md
   # 运行 example1_memory_allocation.py
   ```

2. **第二步：学习基础优化**
   ```bash
   cd optimize_in_gpu/reduce
   # 阅读 README.md（非常详细的优化教程）
   # 对比不同版本的性能
   ```

3. **第三步：实践 CUDA 编程**
   ```bash
   cd universe_best_cuda_practice/1_cuda_reduce_study
   # 从 v0 开始，逐步理解每个优化版本
   ```

### 进阶路径

1. **深入理解 SGEMM 优化**
   - `optimize_in_gpu/sgemm/` - 查看优化版本
   - `universe_best_cuda_practice/2_cuda_sgemm_study/` - 8 个优化版本

2. **学习 Tensor Core**
   - `universe_best_cuda_practice/4_tensor_core_wmma/`
   - `universe_best_cuda_practice/5_mma_and_swizzle/`

3. **掌握性能分析工具**
   - 深入学习 Nsight Systems 的高级功能
   - 使用 SQLite 进行程序化分析

## 📊 性能数据

所有性能数据均在 **NVIDIA V100** GPU 上测试，使用 **Nsight Systems** 进行性能分析。

### 关键性能指标

- **Reduce**: 858 GB/s (95.3% 带宽利用率)
- **SGEMM**: 达到 cuBLAS 96.8% 的性能
- **SGEMV**: 部分场景超过 cuBLAS 性能（110.9%）

## 🛠️ 环境要求

### 必需工具

- **CUDA Toolkit** (>= 10.0)
- **Nsight Systems** (nsys) - GPU 性能分析工具
- **CMake** (>= 3.10) - 用于编译部分示例
- **Python 3** - 用于性能分析示例

### 安装 Nsight Systems

```bash
# 使用项目提供的安装脚本
cd gpu_profile
bash install_nsys_latest.sh

# 或手动安装
# 从 NVIDIA 官网下载并安装
```

## 📝 文档索引

### 快速参考

- [GPU 性能分析快速入门](gpu_profile/QUICK_START.md)
- [Nsight Systems 使用指南](gpu_profile/README_nsys.md)
- [Reduce 优化详细教程](optimize_in_gpu/reduce/README.md)

### 详细文档

- [GPU 性能分析完整指南](gpu_profile/GPU_PROFILING_GUIDE.md)
- [SQLite 查询指南](gpu_profile/SQLITE_QUERY_GUIDE.md)
- [CUDA Timeline 查看方法](gpu_profile/HOW_TO_VIEW_CUDA_TIMELINE.md)
- [故障排除指南](gpu_profile/TROUBLESHOOTING.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如有问题，可以联系：xiandong_liu@foxmail.com

## 📄 许可证

本项目采用 [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) 许可证。

## 🙏 致谢

感谢 NVIDIA 提供的优秀工具和文档支持。

---

**开始你的 GPU 优化之旅吧！** 🚀

