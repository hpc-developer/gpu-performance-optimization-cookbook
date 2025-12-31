"""
示例 4: 内核启动开销问题

问题描述：
----------
大量小内核启动是 GPU 编程中容易被忽视但影响很大的问题。

问题表现：
1. 每次启动内核都有固定开销（参数传递、调度等）
2. 小内核的计算时间可能 < 启动开销
3. GPU 调度器需要频繁调度，增加开销
4. 无法充分利用 GPU 的计算资源

性能影响：
- 启动开销可能占总时间的 50-80%
- 小内核效率 < 50%（启动开销占比高）
- GPU 利用率低（频繁调度）
- 整体吞吐量下降

优化思路：
- 合并小操作为大内核
- 增加每个内核的工作量
- 使用网格级并行而不是循环
- 批量处理数据
"""

import nvtx
import numpy as np
import time

# 尝试导入 GPU 计算库
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        torch.cuda.init()
        DEVICE = torch.device('cuda')
        print(f"✓ PyTorch 可用，使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        TORCH_AVAILABLE = False
        print("⚠ PyTorch 可用但 CUDA 不可用，将使用 CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch 不可用")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print(f"✓ CuPy 可用")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy 不可用")

# 选择可用的 GPU 库
USE_TORCH = TORCH_AVAILABLE
USE_CUPY = CUPY_AVAILABLE and not TORCH_AVAILABLE  # 优先使用 PyTorch

if not (USE_TORCH or USE_CUPY):
    print("⚠ 警告：没有可用的 GPU 库，将使用 CPU 模拟（性能分析可能不准确）")

def get_color(name):
    colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "purple": (0.5, 0.0, 0.5),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def bad_practice_many_small_kernels(size=100, iterations=1000):
    """
    不好的做法：启动大量小内核
    
    问题分析：
    ----------
    这个函数展示了典型的"大量小内核"反模式：
    
    1. 频繁启动小内核（第 1 个问题点）
       - 每次启动都有固定开销：
         * 参数传递和验证（~0.01-0.1ms）
         * GPU 调度器调度（~0.01-0.05ms）
         * 上下文切换（~0.01-0.05ms）
       - 小内核计算时间可能 < 启动开销
    
    2. 效率低（第 2 个问题点）
       - 启动开销：0.1ms
       - 计算时间：0.05ms
       - 效率 = 0.05 / 0.15 = 33%
       - 67% 的时间浪费在启动开销上
    
    3. GPU 利用率低（第 3 个问题点）
       - GPU 频繁在启动和计算之间切换
       - 无法充分利用计算资源
       - 调度开销累积
    
    性能计算：
    - 1000 个小内核：1000 × 0.15ms = 150ms
    - 其中启动开销：1000 × 0.1ms = 100ms（67%）
    - 实际计算：1000 × 0.05ms = 50ms（33%）
    
    在 nsys 时间线中你会看到：
    - 大量短时间的"小内核"标记
    - 内核执行时间很短，但总时间很长
    - GPU 利用率低，频繁调度
    """
    print("=== 不好的做法：大量小内核 ===")
    print("问题：频繁启动小内核，启动开销占比高，效率低\n")
    
    with nvtx.annotate("大量小内核示例", color=get_color("red")):
        for i in range(iterations):
            # ❌ 问题点 1: 频繁启动小内核
            # 每次启动都有固定开销，与内核大小无关
            # 小内核的问题：启动开销可能 > 计算时间
            with nvtx.annotate(f"小内核{i}", color=get_color("orange")):
                # 实际中包括：
                #   - 参数传递和验证：~0.01-0.1ms
                #   - GPU 调度器调度：~0.01-0.05ms
                #   - 上下文切换：~0.01-0.05ms
                # 总计：~0.1ms（固定开销）
                if USE_TORCH:
                    # 小矩阵乘法，计算量小，启动开销占比高
                    a = torch.randn(size, size, device=DEVICE)
                    b = torch.randn(size, size, device=DEVICE)
                    c = torch.matmul(a, b)  # 小内核，启动开销占比高
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    a = cp.random.randn(size, size, dtype=cp.float32)
                    b = cp.random.randn(size, size, dtype=cp.float32)
                    c = cp.matmul(a, b)  # 小内核，启动开销占比高
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.0001)  # 启动开销
                    time.sleep(0.00005)  # 计算时间很短

def good_practice_few_large_kernels(size=100, iterations=1000):
    """
    好的做法：合并为少量大内核
    
    优化分析：
    ----------
    这个函数展示了正确的内核设计方式：
    
    1. 合并操作（优化点 1）
       - 将多个小操作合并为批次
       - 减少内核启动次数
       - 相同的启动开销，处理更多数据
    
    2. 提高效率（优化点 2）
       - 启动开销：0.1ms（与单个小内核相同）
       - 计算时间：100 × 0.05ms = 5ms
       - 效率 = 5 / 5.1 = 98%
       - 启动开销占比 < 2%
    
    3. 提高 GPU 利用率（优化点 3）
       - GPU 可以连续计算，减少调度
       - 充分利用计算资源
       - 提高整体吞吐量
    
    性能计算：
    - 10 个大内核：10 × 5.1ms = 51ms
    - 其中启动开销：10 × 0.1ms = 1ms（2%）
    - 实际计算：10 × 5ms = 50ms（98%）
    - vs 小内核：150ms（启动开销 100ms，67%）
    - 性能提升：150 / 51 ≈ 2.9x
    
    在 nsys 时间线中你会看到：
    - 少量长时间的"大内核"标记
    - 内核执行时间长，启动开销占比低
    - GPU 利用率高，连续执行
    """
    print("=== 好的做法：合并为大内核 ===")
    print("优化：合并小操作为大内核，减少启动次数，提高效率\n")
    
    with nvtx.annotate("合并内核示例", color=get_color("green")):
        # ✅ 优化点 1: 将多个小操作合并为批次
        # 减少内核启动次数：从 N 次减少到 N/batch_size 次
        # 相同的启动开销，处理更多数据
        batch_size = 100
        num_batches = iterations // batch_size
        
        for batch in range(num_batches):
            with nvtx.annotate(f"大内核批次{batch}", color=get_color("blue")):
                # 启动开销相同（~0.1ms），但现在处理更多数据
                # 启动开销占比低，效率高
                if USE_TORCH:
                    # 大矩阵乘法，计算量大，启动开销占比低
                    a = torch.randn(size * 10, size * 10, device=DEVICE)
                    b = torch.randn(size * 10, size * 10, device=DEVICE)
                    c = torch.matmul(a, b)  # 大内核，启动开销占比低
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    a = cp.random.randn(size * 10, size * 10, dtype=cp.float32)
                    b = cp.random.randn(size * 10, size * 10, dtype=cp.float32)
                    c = cp.matmul(a, b)  # 大内核，启动开销占比低
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.0001)  # 启动开销（与单个小内核相同）
                    # ✅ 优化点 2: 批量处理
                    # 在同一个内核中处理多个元素
                    # 利用 GPU 的并行能力（线程、块、网格）
                    for i in range(batch_size):
                        with nvtx.annotate(f"处理元素{batch*batch_size + i}", color=get_color("green")):
                            time.sleep(0.00005)  # 计算时间

def demonstrate_kernel_launch_overhead():
    """演示内核启动开销"""
    print("=== 内核启动开销演示 ===")
    
    with nvtx.annotate("内核启动开销分析", color=get_color("purple")):
        # 小内核：启动开销占比高
        with nvtx.annotate("小内核示例", color=get_color("red")):
            with nvtx.annotate("启动开销", color=get_color("yellow")):
                if USE_TORCH:
                    a = torch.randn(64, 64, device=DEVICE)
                    b = torch.randn(64, 64, device=DEVICE)
                elif USE_CUPY:
                    a = cp.random.randn(64, 64, dtype=cp.float32)
                    b = cp.random.randn(64, 64, dtype=cp.float32)
                else:
                    time.sleep(0.001)  # 启动开销
            with nvtx.annotate("计算时间", color=get_color("blue")):
                if USE_TORCH:
                    c = torch.matmul(a, b)  # 小内核，计算时间短
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    c = cp.matmul(a, b)  # 小内核，计算时间短
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.0005)  # 计算时间短
        
        # 大内核：启动开销占比低
        with nvtx.annotate("大内核示例", color=get_color("green")):
            with nvtx.annotate("启动开销", color=get_color("yellow")):
                if USE_TORCH:
                    a = torch.randn(1024, 1024, device=DEVICE)
                    b = torch.randn(1024, 1024, device=DEVICE)
                elif USE_CUPY:
                    a = cp.random.randn(1024, 1024, dtype=cp.float32)
                    b = cp.random.randn(1024, 1024, dtype=cp.float32)
                else:
                    time.sleep(0.001)  # 相同的启动开销
            with nvtx.annotate("计算时间", color=get_color("blue")):
                if USE_TORCH:
                    c = torch.matmul(a, b)  # 大内核，计算时间长，开销占比低
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    c = cp.matmul(a, b)  # 大内核，计算时间长，开销占比低
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.01)  # 计算时间长，开销占比低

if __name__ == "__main__":
    print("内核启动开销性能分析示例\n")
    
    # 大量小内核
    start = time.time()
    bad_practice_many_small_kernels(iterations=500)
    bad_time = time.time() - start
    print(f"大量小内核耗时: {bad_time:.4f} 秒\n")
    
    # 少量大内核
    start = time.time()
    good_practice_few_large_kernels(iterations=500)
    good_time = time.time() - start
    print(f"合并大内核耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x\n")
    
    # 内核启动开销演示
    demonstrate_kernel_launch_overhead()
    
    print("\n使用 nsys profile 查看内核启动时间线：")
    print("nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --output=example4_kernel_overhead.nsys-rep python example4_kernel_overhead.py")

