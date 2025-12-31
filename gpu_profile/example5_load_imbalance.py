"""
示例 5: 负载不均衡问题

问题描述：
----------
负载不均衡是并行计算中常见但影响很大的问题。

问题表现：
1. 不同任务/线程/GPU 的工作量差异很大
2. 快的任务完成后等待慢的任务
3. 部分资源空闲，部分资源过载
4. 总执行时间由最慢的任务决定

性能影响：
- 资源利用率可能 < 50%
- 总时间 = max(所有任务时间)，而不是平均时间
- 无法充分利用并行性
- 整体效率下降

优化思路：
- 动态负载均衡（工作窃取）
- 根据任务复杂度分配资源
- 使用任务队列管理
- 预测任务执行时间并预分配
"""

import nvtx
import numpy as np
import time
import random

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
        "orange": (1.0, 0.5, 0.0),
        "purple": (0.5, 0.0, 0.5),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def bad_practice_imbalanced_load(num_tasks=8):
    """
    不好的做法：负载不均衡
    
    问题分析：
    ----------
    这个函数展示了典型的"负载不均衡"反模式：
    
    1. 任务负载差异大（第 1 个问题点）
       - 任务负载：0.05, 0.1, 0.15, 0.2, 0.3（差异 6 倍）
       - 快的任务（0.05）很快完成，但需要等待慢的任务（0.3）
       - 资源利用率低
    
    2. 等待时间（第 2 个问题点）
       - 总时间 = max(所有任务时间) = 0.3
       - 如果负载均衡，总时间 ≈ 平均时间 = 0.125
       - 浪费 = 0.3 - 0.125 = 0.175（58% 的时间浪费）
    
    3. 资源浪费（第 3 个问题点）
       - 快的任务完成后资源空闲
       - 无法将空闲资源分配给慢的任务
       - 并行效率低
    
    性能计算：
    - 任务时间：0.05, 0.1, 0.15, 0.2, 0.3, 0.1, 0.05, 0.05
    - 总时间 = max = 0.3（由最慢任务决定）
    - 平均时间 = 0.125
    - 效率 = 0.125 / 0.3 = 42%（58% 浪费）
    
    在 nsys 时间线中你会看到：
    - 任务完成时间差异很大
    - 部分任务很早完成但总时间很长
    - 资源利用率不均匀
    """
    print("=== 不好的做法：负载不均衡 ===")
    print("问题：任务负载差异大，快的任务等待慢的任务，资源浪费\n")
    
    with nvtx.annotate("负载不均衡示例", color=get_color("red")):
        # ❌ 问题点 1: 任务负载差异很大
        # 任务负载范围：不同大小的矩阵（6 倍差异）
        # 这导致快的任务很快完成，但需要等待慢的任务
        task_sizes = [128, 192, 256, 64, 384, 128, 64, 64]  # 不均衡的矩阵大小
        
        for i, size in enumerate(task_sizes[:num_tasks]):
            # 每个任务的工作量不同（不同大小的矩阵乘法）
            # 在并行环境中，这会导致负载不均衡
            load_factor = size / 128.0  # 归一化负载
            with nvtx.annotate(f"任务{i}(负载:{load_factor:.2f},大小:{size})", color=get_color("orange")):
                # 不同大小的GPU矩阵乘法，计算量不同
                if USE_TORCH:
                    a = torch.randn(size, size, device=DEVICE)
                    b = torch.randn(size, size, device=DEVICE)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    a = cp.random.randn(size, size, dtype=cp.float32)
                    b = cp.random.randn(size, size, dtype=cp.float32)
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(load_factor * 0.1)  # 模拟不同负载
        
        # ❌ 问题点 2: 所有任务完成后才能继续
        # 总时间由最慢的任务决定
        # 快的任务完成后资源空闲，无法利用
        with nvtx.annotate("等待所有任务完成", color=get_color("yellow")):
            # 最慢的任务决定了总时间
            # 如果任务负载不均衡，总时间远大于平均时间
            if USE_TORCH:
                torch.cuda.synchronize()
            elif USE_CUPY:
                cp.cuda.Stream.null.synchronize()
            else:
                max_size = max(task_sizes[:num_tasks])
                time.sleep(max_size / 128.0 * 0.1)

def good_practice_balanced_load(num_tasks=8):
    """
    好的做法：负载均衡
    
    优化分析：
    ----------
    这个函数展示了正确的负载分配方式：
    
    1. 平均分配负载（优化点 1）
       - 将总工作量平均分配给所有任务
       - 每个任务的工作量相同
       - 避免任务时间差异
    
    2. 同时完成（优化点 2）
       - 所有任务几乎同时完成
       - 没有等待时间
       - 资源利用率高
    
    3. 性能提升
       - 总时间 = 平均时间 = 0.125
       - vs 不均衡的 0.3
       - 性能提升：0.3 / 0.125 = 2.4x
       - 效率 = 100%（无浪费）
    
    在 nsys 时间线中你会看到：
    - 所有任务时间相同
    - 任务几乎同时完成
    - 资源利用率高且均匀
    """
    print("=== 好的做法：负载均衡 ===")
    print("优化：平均分配负载，所有任务同时完成，无等待时间\n")
    
    with nvtx.annotate("负载均衡示例", color=get_color("green")):
        # ✅ 优化点 1: 将工作负载平均分配
        # 每个任务的工作量相同，避免负载不均衡
        # 实际中可能需要：
        #   - 预测任务执行时间
        #   - 动态调整任务分配
        #   - 使用工作窃取策略
        balanced_size = 256  # 所有任务使用相同大小的矩阵
        
        for i in range(num_tasks):
            # 每个任务负载相同，确保同时完成
            with nvtx.annotate(f"任务{i}(负载:1.0,大小:{balanced_size})", color=get_color("blue")):
                # 每个任务负载相同（相同大小的矩阵乘法）
                # 在并行环境中，这确保所有资源同时完成
                if USE_TORCH:
                    a = torch.randn(balanced_size, balanced_size, device=DEVICE)
                    b = torch.randn(balanced_size, balanced_size, device=DEVICE)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    a = cp.random.randn(balanced_size, balanced_size, dtype=cp.float32)
                    b = cp.random.randn(balanced_size, balanced_size, dtype=cp.float32)
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.125)  # 相同负载
        
        # ✅ 优化点 2: 所有任务几乎同时完成
        # 没有等待时间，资源利用率高
        # 总时间 = 单个任务时间（而不是最慢任务时间）
        with nvtx.annotate("所有任务完成", color=get_color("green")):
            if USE_TORCH:
                torch.cuda.synchronize()
            elif USE_CUPY:
                cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.01)

def demonstrate_work_stealing():
    """演示工作窃取策略"""
    print("=== 工作窃取策略演示 ===")
    
    with nvtx.annotate("工作窃取示例", color=get_color("purple")):
        # 初始任务分配（不同大小的任务）
        task_sizes = [256, 384, 128, 192, 320]
        
        with nvtx.annotate("初始分配", color=get_color("yellow")):
            for i, size in enumerate(task_sizes):
                with nvtx.annotate(f"工作单元{i}(大小:{size})", color=get_color("orange")):
                    if USE_TORCH:
                        a = torch.randn(size, size, device=DEVICE)
                        b = torch.randn(size, size, device=DEVICE)
                        c = torch.matmul(a, b)
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        a = cp.random.randn(size, size, dtype=cp.float32)
                        b = cp.random.randn(size, size, dtype=cp.float32)
                        c = cp.matmul(a, b)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        time.sleep(size / 128.0 * 0.1)
        
        # 快速完成的工作单元可以"窃取"其他单元的工作
        with nvtx.annotate("工作窃取", color=get_color("green")):
            # 模拟负载重新分配（将大任务拆分）
            with nvtx.annotate("重新分配负载", color=get_color("blue")):
                if USE_TORCH:
                    # 将大任务拆分为小任务
                    a = torch.randn(128, 128, device=DEVICE)
                    b = torch.randn(128, 128, device=DEVICE)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    a = cp.random.randn(128, 128, dtype=cp.float32)
                    b = cp.random.randn(128, 128, dtype=cp.float32)
                    c = cp.matmul(a, b)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.05)

if __name__ == "__main__":
    print("负载均衡性能分析示例\n")
    
    # 负载不均衡
    start = time.time()
    bad_practice_imbalanced_load(num_tasks=8)
    bad_time = time.time() - start
    print(f"负载不均衡耗时: {bad_time:.4f} 秒\n")
    
    # 负载均衡
    start = time.time()
    good_practice_balanced_load(num_tasks=8)
    good_time = time.time() - start
    print(f"负载均衡耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x\n")
    
    # 工作窃取演示
    demonstrate_work_stealing()
    
    print("\n使用 nsys profile 查看负载分布：")
    print("nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --output=example5_load_imbalance.nsys-rep python example5_load_imbalance.py")

