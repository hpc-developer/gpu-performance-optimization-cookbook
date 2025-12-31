# 如何在 Nsight Systems GUI 中查看 CUDA Timeline

## 验证 CUDA 数据是否存在

从命令行验证 CUDA 数据：

```bash
# 检查 CUDA API 调用
nsys stats --report=cuda_api_sum example1_basic_profile.nsys-rep

# 检查 CUDA GPU Kernels
nsys stats --report=cuda_gpu_kern_sum example1_basic_profile.nsys-rep
```

如果这些命令有输出，说明 CUDA 数据已经正确记录。

## 在 GUI 中查看 CUDA Timeline

### 方法 1: 基本步骤

1. **打开 .nsys-rep 文件**
   - 启动 Nsight Systems GUI
   - File -> Open -> 选择 `.nsys-rep` 文件

2. **查看 Timeline 视图**
   - 默认应该显示 Timeline 视图
   - 在左侧面板中，应该能看到以下行：
     - **CUDA API** - 显示所有 CUDA API 调用
     - **CUDA Kernels** - 显示所有 GPU kernel 执行
     - **GPU** - 显示 GPU 活动（如果启用了 GPU metrics）

3. **如果看不到 CUDA 行**
   - 检查 View -> Timeline 菜单
   - 确保 "Show CUDA" 或相关选项被勾选
   - 右键点击 timeline 区域 -> Show/Hide Rows
   - 确保 CUDA API 和 CUDA Kernels 行被勾选显示

### 方法 2: 使用 View 菜单

1. **View -> Timeline**
   - 确保 Timeline 视图已激活

2. **View -> Timeline -> Show Rows**
   - 勾选 "CUDA API"
   - 勾选 "CUDA Kernels"
   - 勾选 "GPU" (如果可用)

### 方法 3: 右键菜单

1. 在 Timeline 视图的左侧行列表中右键点击
2. 选择 "Show/Hide Rows"
3. 在弹出的对话框中勾选所有 CUDA 相关的行

## 常见问题

### Q: 为什么看不到 CUDA timeline？

**A: 可能的原因：**

1. **没有使用 --trace=cuda 参数**
   - 检查生成报告的脚本是否包含 `--trace=cuda`
   - 所有示例脚本现在都已包含此参数

2. **GUI 中的行被隐藏**
   - 按照上面的方法 2 或 3 显示 CUDA 行

3. **没有实际的 CUDA 操作**
   - 确保 Python 代码确实执行了 GPU 操作
   - 检查是否有 CUDA 错误（查看 Python 输出）

### Q: 如何确认 CUDA 数据已记录？

**A: 使用命令行验证：**

```bash
# 查看 CUDA API 摘要
nsys stats --report=cuda_api_sum your_file.nsys-rep

# 查看 CUDA Kernel 摘要
nsys stats --report=cuda_gpu_kern_sum your_file.nsys-rep
```

如果有输出，说明数据已记录。

### Q: Timeline 中只看到 NVTX 标记，没有 CUDA？

**A: 这可能是正常的**

- NVTX 标记是应用层标记，用于标识代码区域
- CUDA API 和 Kernels 是实际的 GPU 操作
- 两者应该同时显示在 timeline 中
- 如果只看到 NVTX，检查：
  1. Python 代码是否真的执行了 GPU 操作
  2. 是否使用了 `--trace=cuda` 参数
  3. 是否有 CUDA 错误阻止了 GPU 操作

## 示例：查看 example1_basic_profile.nsys-rep

```bash
# 1. 验证数据存在
nsys stats --report=cuda_api_sum example1_basic_profile.nsys-rep
nsys stats --report=cuda_gpu_kern_sum example1_basic_profile.nsys-rep

# 2. 在 GUI 中打开
nsys-ui example1_basic_profile.nsys-rep
# 或者直接双击 .nsys-rep 文件
```

在 GUI 中，你应该能看到：
- **NVTX 标记**（如 "GPU: 矩阵乘法"）
- **CUDA API 调用**（如 cudaLaunchKernel, cudaMalloc）
- **CUDA Kernels**（如 cutlass::Kernel2, at::native::...）

## 提示

- CUDA timeline 通常显示在 timeline 视图的下方
- 可以缩放 timeline 来查看详细信息
- 点击 CUDA API 调用可以看到详细的参数信息
- 点击 CUDA Kernel 可以看到 kernel 的执行时间和参数

