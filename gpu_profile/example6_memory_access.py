"""
示例 6: 内存访问模式问题

问题描述：
----------
内存访问模式对 GPU 性能有巨大影响，是容易被忽视的性能问题。

问题表现：
1. 非连续内存访问（跨步访问）导致缓存命中率低
2. 内存带宽利用率低（无法充分利用合并访问）
3. 访问延迟增加（需要多次内存事务）
4. 整体性能下降

性能影响：
- 跨步访问的带宽可能只有连续访问的 10-20%
- 缓存命中率可能 < 10%（vs 连续访问的 > 90%）
- 访问延迟增加 5-10 倍
- 整体性能下降 5-10 倍

优化思路：
- 使用连续内存布局（row-major）
- 合并内存访问（coalesced access）
- 使用共享内存缓存
- 数据重排优化访问模式
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
        "orange": (1.0, 0.5, 0.0),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def bad_practice_strided_access(size=1000):
    """
    不好的做法：跨步访问（非连续）
    
    问题分析：
    ----------
    这个函数展示了典型的"跨步访问"反模式：
    
    1. 按列访问（第 1 个问题点）
       - 数据在内存中是按行存储的（row-major）
       - 按列访问需要跨步：访问 data[0, col], data[1, col], ...
       - 每次访问间隔 size 个元素（stride = size）
    
    2. 缓存不友好（第 2 个问题点）
       - 跨步访问导致缓存命中率低
       - 每次访问可能都需要从内存读取
       - 无法利用缓存局部性
    
    3. 带宽利用率低（第 3 个问题点）
       - 无法合并访问（coalesced access）
       - 需要多次内存事务
       - 带宽利用率可能 < 20%
    
    性能影响：
    - 连续访问：带宽利用率 ~80-90%，延迟 ~100ns
    - 跨步访问：带宽利用率 ~10-20%，延迟 ~500-1000ns
    - 性能下降：5-10 倍
    
    在 nsys 时间线中你会看到：
    - "按列访问"标记持续时间长
    - GPU Metrics 中内存带宽利用率低
    - 内存事务数量多但带宽低
    """
    print("=== 不好的做法：跨步内存访问 ===")
    print("问题：按列访问导致跨步访问，缓存命中率低，带宽利用率低\n")
    
    with nvtx.annotate("跨步访问示例", color=get_color("red")):
        # 创建GPU数据
        with nvtx.annotate("创建GPU数据", color=get_color("blue")):
            if USE_TORCH:
                data = torch.randn(size, size, device=DEVICE)
                torch.cuda.synchronize()
            elif USE_CUPY:
                data = cp.random.randn(size, size, dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()
            else:
                data = np.random.rand(size, size).astype(np.float32)
        
        # ❌ 问题点 1: 按列访问（跨步访问）
        # 数据在内存中是按行存储的：row0, row1, row2, ...
        # 按列访问：data[0, col], data[1, col], data[2, col], ...
        # 每次访问间隔 size 个元素（stride = size）
        # 这导致：
        #   - 缓存命中率低（每次访问可能都不在缓存中）
        #   - 无法合并访问（GPU 无法合并多个线程的访问）
        #   - 需要多次内存事务
        with nvtx.annotate("按列访问(跨步)", color=get_color("orange")):
            result = 0.0
            for col in range(min(size, 100)):  # 限制迭代次数
                with nvtx.annotate(f"列{col}", color=get_color("yellow")):
                    # 跨步访问：访问 data[:, col]
                    # 内存访问模式：data[0,col], data[1,col], data[2,col], ...
                    # 每次访问间隔 size 个元素
                    # 缓存不友好，带宽利用率低
                    if USE_TORCH:
                        column_sum = torch.sum(data[:, col])
                        result += column_sum.item()
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        column_sum = cp.sum(data[:, col])
                        result += float(column_sum)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        column_sum = np.sum(data[:, col])
                        result += column_sum
                        time.sleep(0.0001)  # 跨步访问延迟高

def good_practice_contiguous_access(size=1000):
    """
    好的做法：连续内存访问
    
    优化分析：
    ----------
    这个函数展示了正确的内存访问方式：
    
    1. 按行访问（优化点 1）
       - 数据在内存中是按行存储的
       - 按行访问是连续的：data[row, 0], data[row, 1], ...
       - 每次访问相邻元素（stride = 1）
    
    2. 缓存友好（优化点 2）
       - 连续访问导致缓存命中率高
       - 可以利用缓存局部性
       - 减少内存访问次数
    
    3. 带宽利用率高（优化点 3）
       - 可以合并访问（coalesced access）
       - GPU 可以合并多个线程的访问为一次内存事务
       - 带宽利用率高（80-90%）
    
    性能提升：
    - 连续访问：带宽利用率 ~80-90%，延迟 ~100ns
    - vs 跨步访问：带宽利用率 ~10-20%，延迟 ~500-1000ns
    - 性能提升：5-10 倍
    
    在 nsys 时间线中你会看到：
    - "按行访问"标记持续时间短
    - GPU Metrics 中内存带宽利用率高
    - 内存事务数量少但带宽高
    """
    print("=== 好的做法：连续内存访问 ===")
    print("优化：按行访问实现连续访问，缓存命中率高，带宽利用率高\n")
    
    with nvtx.annotate("连续访问示例", color=get_color("green")):
        # 创建GPU数据
        with nvtx.annotate("创建GPU数据", color=get_color("blue")):
            if USE_TORCH:
                data = torch.randn(size, size, device=DEVICE)
                torch.cuda.synchronize()
            elif USE_CUPY:
                data = cp.random.randn(size, size, dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()
            else:
                data = np.random.rand(size, size).astype(np.float32)
        
        # ✅ 优化点 1: 按行访问（连续访问）
        # 数据在内存中是按行存储的：row0, row1, row2, ...
        # 按行访问：data[row, 0], data[row, 1], data[row, 2], ...
        # 每次访问相邻元素（stride = 1）
        # 这导致：
        #   - 缓存命中率高（相邻元素可能在同一个缓存行中）
        #   - 可以合并访问（GPU 可以合并多个线程的访问）
        #   - 需要更少的内存事务
        with nvtx.annotate("按行访问(连续)", color=get_color("blue")):
            result = 0.0
            for row in range(min(size, 100)):  # 限制迭代次数
                with nvtx.annotate(f"行{row}", color=get_color("green")):
                    # 连续访问：访问 data[row, :]
                    # 内存访问模式：data[row,0], data[row,1], data[row,2], ...
                    # 每次访问相邻元素
                    # 缓存友好，带宽利用率高
                    if USE_TORCH:
                        row_sum = torch.sum(data[row, :])
                        result += row_sum.item()
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        row_sum = cp.sum(data[row, :])
                        result += float(row_sum)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        row_sum = np.sum(data[row, :])
                        result += row_sum
                        time.sleep(0.00005)  # 连续访问延迟低

def demonstrate_cache_efficiency():
    """演示缓存效率差异"""
    print("=== 缓存效率演示 ===")
    
    size = 1000
    with nvtx.annotate("缓存效率分析", color=get_color("blue")):
        # 创建GPU数据
        with nvtx.annotate("创建GPU数据", color=get_color("yellow")):
            if USE_TORCH:
                data = torch.randn(size, size, device=DEVICE)
                torch.cuda.synchronize()
            elif USE_CUPY:
                data = cp.random.randn(size, size, dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()
            else:
                data = np.random.rand(size, size).astype(np.float32)
        
        # 连续访问模式
        with nvtx.annotate("连续访问(高缓存命中)", color=get_color("green")):
            # 访问连续内存块
            for i in range(0, min(size, 100), 10):
                with nvtx.annotate(f"块{i//10}", color=get_color("blue")):
                    if USE_TORCH:
                        chunk = data[i:i+10, :]
                        result = torch.sum(chunk)
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        chunk = data[i:i+10, :]
                        result = cp.sum(chunk)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        chunk = data[i:i+10, :]
                        np.sum(chunk)
                        time.sleep(0.001)
        
        # 随机访问模式
        with nvtx.annotate("随机访问(低缓存命中)", color=get_color("red")):
            # 随机访问，缓存命中率低
            indices = np.random.permutation(min(size, 100))
            for idx in indices[:20]:
                with nvtx.annotate(f"随机索引{idx}", color=get_color("orange")):
                    if USE_TORCH:
                        result = torch.sum(data[idx, :])
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        result = cp.sum(data[idx, :])
                        cp.cuda.Stream.null.synchronize()
                    else:
                        np.sum(data[idx, :])
                        time.sleep(0.002)  # 更慢的访问

if __name__ == "__main__":
    print("内存访问模式性能分析示例\n")
    
    # 跨步访问
    start = time.time()
    bad_practice_strided_access(size=500)
    bad_time = time.time() - start
    print(f"跨步访问耗时: {bad_time:.4f} 秒\n")
    
    # 连续访问
    start = time.time()
    good_practice_contiguous_access(size=500)
    good_time = time.time() - start
    print(f"连续访问耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x\n")
    
    # 缓存效率演示
    demonstrate_cache_efficiency()
    
    print("\n使用 nsys profile 查看内存访问模式：")
    print("nsys profile --trace=cuda,nvtx --cuda-memory-usage=true --output=example6_memory_access.nsys-rep python example6_memory_access.py")

