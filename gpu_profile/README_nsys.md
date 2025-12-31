# Nsight Systems (nsys) Profile 示例集合

这个目录包含多个示例脚本，展示 `nsys profile` 命令的不同参数和用法。

## 文件说明

### 示例脚本
- `profile_example1_memory_allocation.sh` - GPU 内存分配问题分析
- `profile_example2_data_transfer.sh` - CPU-GPU 数据传输瓶颈分析
- `profile_example3_synchronization.sh` - GPU 同步问题分析
- `profile_example4_kernel_overhead.sh` - GPU Kernel 启动开销分析
- `profile_example5_load_imbalance.sh` - GPU 负载不均衡问题分析
- `profile_example6_memory_access.sh` - GPU 内存访问模式问题分析
- `profile_example7_comprehensive.sh` - 综合性能问题分析
- `example10_comprehensive.sh` - 综合完整示例

### 运行脚本
- `run_all_examples.sh` - 运行所有示例说明
- `nsys_profile_examples.sh` - 主脚本框架

## 快速开始

### 1. 查看所有示例说明
```bash
bash run_all_examples.sh
```

### 2. 运行单个示例
```bash
bash profile_example1_memory_allocation.sh
```

### 3. 实际执行分析（需要取消注释）
编辑示例脚本，取消注释实际执行命令，例如：
```bash
nsys profile --trace=cuda,nvtx --output=nsys_profiles/basic_profile.nsys-rep python3 example1_memory_allocation.py
```

## 常用命令速查

### 基本用法
```bash
# 最简单的用法
nsys profile --output=output.nsys-rep python script.py

# 跟踪 CUDA + NVTX（推荐）
nsys profile --trace=cuda,nvtx --output=output.nsys-rep python script.py
```

### 内存分析
```bash
# 启用内存跟踪
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=output.nsys-rep python script.py
```

### 高频率采样
```bash
# 高频率采样（更详细但文件更大）
nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --gpu-metrics-frequency=100 --output=output.nsys-rep python script.py
```

### 时间控制
```bash
# 只分析前 10 秒
nsys profile --duration=10 --output=output.nsys-rep python script.py

# 等待 5 秒后分析 10 秒
nsys profile --wait=5 --duration=10 --output=output.nsys-rep python script.py
```

### 导出格式
```bash
# 导出为 SQLite 和 JSON
nsys profile --export=sqlite,json --output=output.nsys-rep python script.py

# 导出为 Chrome 跟踪格式
nsys profile --export=chrome-trace --output=output.nsys-rep python script.py
```

### 完整配置（推荐用于生产环境）
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --sampling-frequency=1000 \
  --gpu-metrics-frequency=100 \
  --stats=true \
  --force-overwrite=true \
  --output=profile.nsys-rep \
  python script.py
```

## 查看结果

### 使用 Nsight Systems GUI
```bash
nsys-ui profile.nsys-rep
# 或
nsight-sys profile.nsys-rep
```

### 命令行查看统计信息
```bash
nsys stats profile.nsys-rep
```

### 导出报告
```bash
# 导出为 SQLite（已包含在 --export 中）
# 或使用 stats 命令
nsys stats --report gputrace profile.nsys-rep
```

## 参数说明

### 主要参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--trace` | 跟踪的组件 | `cuda`, `nvtx`, `osrt`, `opengl`, `vulkan` |
| `--output` | 输出文件名 | `output.nsys-rep` |
| `--cuda-memory-usage` | 跟踪 CUDA 内存 | `true`, `false` |
| `--sampling-frequency` | CPU 采样频率 (Hz) | `100`, `1000` |
| `--gpu-metrics-frequency` | GPU 指标采样频率 (Hz) | `10`, `100` |
| `--duration` | 分析持续时间 (秒) | `10`, `60` |
| `--wait` | 等待时间 (秒) | `5`, `10` |
| `--stats` | 显示统计信息 | `true`, `false` |
| `--export` | 导出格式 | `sqlite`, `json`, `chrome-trace` |
| `--force-overwrite` | 强制覆盖输出文件 | `true`, `false` |
| `--attach` | 附加到进程 (PID) | `<进程ID>` |

### 跟踪选项说明

- `cuda`: CUDA API 调用（默认包含）
- `nvtx`: NVTX 标记（推荐与 NVTX 代码一起使用）
- `osrt`: 操作系统运行时（系统调用等）
- `opengl`: OpenGL API 调用
- `vulkan`: Vulkan API 调用

## 最佳实践

1. **开发阶段**: 使用 `--trace=cuda,nvtx` 和较低的采样频率
2. **性能分析**: 使用高采样频率和内存跟踪
3. **长时间运行**: 使用 `--duration` 限制分析时间
4. **内存问题**: 启用 `--cuda-memory-usage=true`
5. **生产环境**: 使用完整配置，包含所有相关跟踪选项

## 注意事项

- 采样频率越高，生成的文件越大，分析时间越长
- 某些跟踪选项可能不适用于所有应用（如 opengl, vulkan）
- 使用 `--force-overwrite` 时要小心，避免覆盖重要数据
- 附加到进程时，确保有足够的权限

## 相关资源

- [NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)
- [Nsight Systems 用户指南](https://developer.nvidia.com/nsight-systems)
- [NVTX 标记文档](https://nvidia.github.io/NVTX/)

