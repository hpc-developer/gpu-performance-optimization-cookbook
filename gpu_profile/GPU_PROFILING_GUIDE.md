# GPU 性能分析指南

## 概述

示例脚本（如 `example1_memory_allocation.py`）包含实际的 GPU 计算，可用于真实的 CUDA 性能分析。

## 功能特性

### 支持的 GPU 库
- **PyTorch**（优先使用）- 如果可用且 CUDA 可用
- **CuPy** - 如果 PyTorch 不可用
- **CPU 回退** - 如果没有 GPU 库，使用 CPU 模拟（性能分析不准确）

### GPU 计算类型

1. **矩阵乘法** - 使用 `gpu_matrix_multiply()` 函数
2. **数据传输** - CPU ↔ GPU 数据传输
3. **数据预处理** - 归一化、reshape 等操作
4. **深度学习操作** - 前向传播、反向传播、参数更新
5. **向量运算** - 求和、平方等

## 使用方法

### 1. 基本运行

```bash
conda activate python3.12
python3 example1_memory_allocation.py
```

### 2. 使用 nsys profile 收集性能数据

```bash
# 基本分析
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=gpu_profile.nsys-rep \
  python3 example1_memory_allocation.py

# 详细分析（推荐）
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sampling-frequency=1000 \
  --gpu-metrics-frequency=100 \
  --export=sqlite \
  --output=gpu_profile.nsys-rep \
  python3 example1_memory_allocation.py
```

### 3. 查看分析结果

```bash
# 使用 GUI 查看时间线
nsys-ui gpu_profile.nsys-rep

# 查看统计信息
nsys stats gpu_profile.nsys-rep

# 查询 SQLite 数据
./query_nvtx_data.sh gpu_profile.sqlite
```

## 代码结构

### GPU 计算函数

```python
def gpu_matrix_multiply(size=1024):
    """GPU 矩阵乘法计算"""
    # 支持 PyTorch 和 CuPy
    # 包含 NVTX 标记用于性能分析
```

### 主要示例函数

所有示例函数现在都包含实际的 GPU 计算：

- `demo_basic_range()` - 基本 GPU 矩阵乘法
- `demo_range_with_label()` - GPU 数据预处理
- `demo_range_with_color()` - 多个 GPU 计算
- `demo_domain()` - GPU 数据传输和计算
- `demo_nested_ranges()` - 嵌套的 GPU 操作
- `demo_deeply_nested_ranges()` - 深层嵌套 GPU 计算
- `demo_mixed_nested_with_domains()` - 混合 GPU 操作
- `demo_real_world_example()` - 完整的深度学习训练流程

## 性能分析要点

### 1. CUDA 内核分析

在 Nsight Systems GUI 中，你可以看到：
- **GPU Kernels** - 所有 CUDA 内核的执行时间
- **Kernel 启动开销** - 内核启动和调度时间
- **GPU 利用率** - SM 利用率和内存带宽利用率

### 2. 内存分析

- **GPU Memory** - 内存分配和释放
- **数据传输** - CPU ↔ GPU 数据传输时间
- **内存带宽** - 内存访问效率

### 3. NVTX 标记分析

- **时间线视图** - 查看各个 GPU 操作的执行时间
- **嵌套结构** - 理解操作的层次关系
- **性能瓶颈** - 识别最耗时的操作

## 示例分析场景

### 场景 1: 矩阵乘法性能

```bash
# 运行分析
nsys profile --trace=cuda,nvtx --output=matmul_profile.nsys-rep python3 example1_memory_allocation.py

# 在 GUI 中查看
# 1. 找到 "GPU: 矩阵乘法" 标记
# 2. 查看对应的 CUDA 内核执行时间
# 3. 分析内存带宽利用率
```

### 场景 2: 数据传输瓶颈

```bash
# 运行分析
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=transfer_profile.nsys-rep python3 example2_data_transfer.py

# 在 GUI 中查看
# 1. 找到 "GPU: CPU->GPU 传输" 标记
# 2. 查看数据传输时间
# 3. 分析传输效率
```

### 场景 3: 深度学习训练流程

```bash
# 运行分析
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=training_profile.nsys-rep python3 example7_comprehensive.py

# 在 GUI 中查看
# 1. 查看 "训练循环" 的整体结构
# 2. 分析前向传播、反向传播的时间分布
# 3. 识别性能瓶颈
```

## 性能优化建议

### 1. 减少数据传输

- 批量传输数据，而不是频繁小传输
- 使用 pinned memory 提高传输速度
- 重叠计算和传输（使用 CUDA streams）

### 2. 优化内核启动

- 合并小操作为大内核
- 减少内核启动次数
- 使用动态并行（如果适用）

### 3. 提高内存效率

- 使用连续内存访问
- 优化数据布局
- 使用共享内存缓存

### 4. 提高 GPU 利用率

- 确保 GPU 持续工作，减少空闲时间
- 使用多个 CUDA streams 并行执行
- 平衡计算和内存访问

## 常见问题

### Q: 为什么看不到 GPU 计算？

**A:** 检查：
1. 是否使用了 `--trace=cuda` 参数
2. GPU 是否可用（`nvidia-smi`）
3. PyTorch/CuPy 是否正确安装

### Q: 如何查看具体的 CUDA 内核？

**A:** 在 Nsight Systems GUI 中：
1. 展开 "GPU Kernels" 时间线
2. 点击具体的内核查看详细信息
3. 查看内核参数和执行时间

### Q: 如何分析内存使用？

**A:** 
1. 使用 `--cuda-memory-usage=true` 参数
2. 在 GUI 中查看 "GPU Memory" 时间线
3. 分析内存分配和释放模式

## 进一步学习

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch 性能调优](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

