"""
示例 3: 不必要的同步问题

问题描述：
----------
频繁的同步操作是 GPU 编程中另一个常见的性能杀手。

问题表现：
1. 每次操作后立即同步，等待 GPU 完成
2. 同步操作阻塞 CPU，无法继续提交新任务
3. GPU 流水线被频繁打断，无法充分利用并行性
4. CPU 和 GPU 无法并行工作

性能影响：
- 同步时间可能占总时间的 40-60%
- GPU 利用率低（等待同步）
- 无法利用异步执行的并行性
- 整体吞吐量下降

优化思路：
- 延迟同步，批量处理结果
- 使用异步流（streams）管理并发
- 只在真正需要结果时同步
- 使用事件（events）进行细粒度同步
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

def bad_practice_frequent_sync(iterations=100):
    """
    不好的做法：每次操作后都同步
    
    问题分析：
    ----------
    这个函数展示了典型的"频繁同步"反模式：
    
    1. 启动后立即同步（第 1 个问题点）
       - 每次启动 GPU 操作后立即同步
       - 同步操作会阻塞 CPU，等待 GPU 完成
       - 无法利用 GPU 的异步执行能力
    
    2. 流水线阻塞（第 2 个问题点）
       - 同步打断了 GPU 流水线
       - GPU 无法并行处理多个操作
       - CPU 和 GPU 无法重叠工作
    
    3. 性能影响
       - 每次同步需要等待 GPU 完成（~0.5-2ms）
       - 100 次同步 = 50-200ms 的等待时间
       - GPU 利用率可能 < 30%
    
    时间线分析：
    - 启动计算（1ms）→ 同步等待（2ms）→ 获取结果
    - GPU 在同步期间可能空闲
    - 总时间 = 100 × (1 + 2) = 300ms
    
    在 nsys 时间线中你会看到：
    - 大量"同步等待"标记，占用大量时间
    - GPU 利用率低，有很多空闲时间
    - 启动和同步交替出现，无法并行
    """
    print("=== 不好的做法：频繁同步 ===")
    print("问题：每次操作后立即同步，阻塞流水线，无法并行\n")
    
    with nvtx.annotate("频繁同步示例", color=get_color("red")):
        results = []
        
        for i in range(iterations):
            # 启动异步 GPU 操作
            with nvtx.annotate(f"迭代{i}: 启动GPU计算", color=get_color("blue")):
                if USE_TORCH:
                    x = torch.randn(256, 256, device=DEVICE)
                    y = torch.randn(256, 256, device=DEVICE)
                    z = torch.matmul(x, y)  # 异步操作，不阻塞
                elif USE_CUPY:
                    x = cp.random.randn(256, 256, dtype=cp.float32)
                    y = cp.random.randn(256, 256, dtype=cp.float32)
                    z = cp.matmul(x, y)  # 异步操作，不阻塞
                else:
                    time.sleep(0.001)  # 模拟异步操作
            
            # ❌ 问题点 1: 立即同步等待
            # 同步操作会阻塞 CPU，等待 GPU 完成当前操作
            # 这打断了 GPU 流水线，无法并行处理多个操作
            # 如果 GPU 操作很快，同步开销可能比计算时间还长
            with nvtx.annotate(f"迭代{i}: 同步等待", color=get_color("red")):
                # 实际中 cudaDeviceSynchronize() 或 cudaStreamSynchronize()
                # 同步时间 = GPU 执行时间 + 同步开销（~0.1-0.5ms）
                if USE_TORCH:
                    torch.cuda.synchronize()  # 阻塞等待GPU完成
                elif USE_CUPY:
                    cp.cuda.Stream.null.synchronize()  # 阻塞等待GPU完成
                else:
                    time.sleep(0.002)  # 模拟同步开销
            
            # 获取结果（同步后才能安全访问）
            with nvtx.annotate(f"迭代{i}: 获取结果", color=get_color("yellow")):
                if USE_TORCH or USE_CUPY:
                    results.append(z.sum().item() if USE_TORCH else float(cp.sum(z)))
                else:
                    results.append(i)

def good_practice_deferred_sync(iterations=100):
    """
    好的做法：延迟同步，批量处理
    
    优化分析：
    ----------
    这个函数展示了正确的同步策略：
    
    1. 批量启动操作（优化点 1）
       - 连续启动所有异步操作
       - GPU 可以并行处理多个操作
       - CPU 可以继续做其他事情
    
    2. 延迟同步（优化点 2）
       - 只在最后同步一次
       - 让 GPU 有足够时间并行执行
       - 减少同步次数和总等待时间
    
    3. 批量获取结果（优化点 3）
       - 同步后一次性获取所有结果
       - 避免多次同步
    
    性能提升：
    - 启动所有操作：100 × 0.001ms = 0.1ms（非阻塞）
    - 最终同步：0.002ms（只一次）
    - 总时间 ≈ 0.1 + 0.002 = 0.102ms（vs 之前的 300ms）
    - 性能提升：300 / 0.102 ≈ 2941x（理论值）
    
    在 nsys 时间线中你会看到：
    - 大量"启动任务"标记连续出现（无同步）
    - 只有一个"最终同步"标记
    - GPU 利用率高，连续执行
    """
    print("=== 好的做法：延迟同步 ===")
    print("优化：批量启动操作，延迟同步，提高并行性\n")
    
    with nvtx.annotate("延迟同步示例", color=get_color("green")):
        # ✅ 优化点 1: 批量启动所有异步操作
        # 连续启动，不等待完成
        # GPU 可以并行处理多个操作（如果有多个流）
        # CPU 可以继续做其他事情，不需要等待
        with nvtx.annotate("批量启动GPU计算", color=get_color("blue")):
            gpu_results = []
            for i in range(iterations):
                with nvtx.annotate(f"启动任务{i}", color=get_color("green")):
                    # 实际中：cudaLaunchKernel() 立即返回
                    # GPU 在后台执行，CPU 可以继续
                    if USE_TORCH:
                        x = torch.randn(256, 256, device=DEVICE)
                        y = torch.randn(256, 256, device=DEVICE)
                        z = torch.matmul(x, y)  # 异步，不阻塞
                        gpu_results.append(z)
                    elif USE_CUPY:
                        x = cp.random.randn(256, 256, dtype=cp.float32)
                        y = cp.random.randn(256, 256, dtype=cp.float32)
                        z = cp.matmul(x, y)  # 异步，不阻塞
                        gpu_results.append(z)
                    else:
                        time.sleep(0.001)  # 模拟异步操作
        
        # ✅ 优化点 2: 只在最后同步一次
        # 让 GPU 有足够时间并行执行所有操作
        # 只等待一次，而不是 N 次
        # 如果使用多个流，甚至可以完全避免同步
        with nvtx.annotate("最终同步", color=get_color("orange")):
            # 实际中：cudaDeviceSynchronize() 或 cudaStreamSynchronize()
            # 等待所有操作完成
            if USE_TORCH:
                torch.cuda.synchronize()  # 只同步一次
            elif USE_CUPY:
                cp.cuda.Stream.null.synchronize()  # 只同步一次
            else:
                time.sleep(0.002)  # 模拟同步
        
        # ✅ 优化点 3: 批量获取结果
        # 同步后一次性获取所有结果
        # 避免多次同步和多次数据传输
        with nvtx.annotate("批量获取结果", color=get_color("yellow")):
            if USE_TORCH:
                results = [z.sum().item() for z in gpu_results]
            elif USE_CUPY:
                results = [float(cp.sum(z)) for z in gpu_results]
            else:
                results = list(range(iterations))

def demonstrate_pipeline_stall():
    """演示流水线阻塞问题"""
    print("=== 流水线阻塞演示 ===")
    
    with nvtx.annotate("流水线示例", color=get_color("blue")):
        # 任务 1
        with nvtx.annotate("任务1: CPU->GPU传输", color=get_color("yellow")):
            if USE_TORCH:
                cpu_data = torch.randn(512, 512)
                gpu_data = cpu_data.to(DEVICE)
            elif USE_CUPY:
                cpu_data = np.random.randn(512, 512).astype(np.float32)
                gpu_data = cp.asarray(cpu_data)
            else:
                time.sleep(0.01)
        
        # 不好的做法：立即同步
        with nvtx.annotate("同步点(阻塞)", color=get_color("red")):
            if USE_TORCH:
                torch.cuda.synchronize()
            elif USE_CUPY:
                cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.005)
        
        # 任务 2
        with nvtx.annotate("任务2: GPU计算", color=get_color("green")):
            if USE_TORCH:
                result = torch.matmul(gpu_data, gpu_data)
            elif USE_CUPY:
                result = cp.matmul(gpu_data, gpu_data)
            else:
                time.sleep(0.01)
        
        # 又同步
        with nvtx.annotate("同步点(阻塞)", color=get_color("red")):
            if USE_TORCH:
                torch.cuda.synchronize()
            elif USE_CUPY:
                cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.005)
        
        # 任务 3
        with nvtx.annotate("任务3: GPU->CPU传输", color=get_color("yellow")):
            if USE_TORCH:
                cpu_result = result.cpu()
            elif USE_CUPY:
                cpu_result = cp.asnumpy(result)
            else:
                time.sleep(0.01)

if __name__ == "__main__":
    print("同步性能分析示例\n")
    
    # 频繁同步
    start = time.time()
    bad_practice_frequent_sync(iterations=50)
    bad_time = time.time() - start
    print(f"频繁同步耗时: {bad_time:.4f} 秒\n")
    
    # 延迟同步
    start = time.time()
    good_practice_deferred_sync(iterations=50)
    good_time = time.time() - start
    print(f"延迟同步耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x\n")
    
    # 流水线阻塞演示
    demonstrate_pipeline_stall()
    
    print("\n使用 nsys profile 查看同步点：")
    print("nsys profile --trace=cuda,nvtx --sampling-frequency=1000 --output=example3_synchronization.nsys-rep python example3_synchronization.py")

