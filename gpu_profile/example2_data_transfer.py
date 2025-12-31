"""
示例 2: CPU-GPU 数据传输瓶颈

问题描述：
----------
频繁的小数据传输是 GPU 编程中最常见的性能瓶颈之一。

问题表现：
1. 每次迭代都进行 CPU→GPU 和 GPU→CPU 传输
2. 小数据传输的固定开销占比高（启动开销、同步开销）
3. PCIe 总线利用率低（小传输无法充分利用带宽）
4. CPU 和 GPU 相互等待，无法并行

性能影响：
- 传输时间可能占总时间的 30-70%
- 小传输的效率通常 < 10%（固定开销大）
- GPU 计算能力无法充分利用（等待数据传输）

优化思路：
- 批量准备数据，一次性传输
- 使用异步传输和流水线
- 尽量减少不必要的回传
- 使用 pinned memory 提高传输速度
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

def bad_practice_small_transfers(size=1000, iterations=100):
    """
    不好的做法：频繁的小数据传输
    
    问题分析：
    ----------
    这个函数展示了典型的"频繁小传输"反模式：
    
    1. 循环中的数据传输（第 1 个问题点）
       - 每次迭代都进行 CPU→GPU 传输
       - 小数据传输的固定开销占比高：
         * 启动传输操作的开销（~0.1ms）
         * PCIe 总线延迟（~0.1-0.5ms）
         * CPU-GPU 同步开销
    
    2. 立即回传结果（第 2 个问题点）
       - 每次计算后立即传回 CPU
       - 如果不需要立即使用结果，这是浪费
       - 阻塞 GPU 继续计算
    
    3. 无法并行（第 3 个问题点）
       - CPU 准备数据 → 等待传输完成 → GPU 计算 → 等待回传
       - 无法重叠计算和传输
       - 资源利用率低
    
    性能计算：
    - 小传输（1KB）: 固定开销 ~0.5ms，有效传输 ~0.001ms
    - 效率 = 0.001 / 0.501 ≈ 0.2%
    - 100 次传输 = 100 × 0.501ms = 50.1ms
    
    在 nsys 时间线中你会看到：
    - 大量短时间的"CPU->GPU 传输"标记
    - 传输操作占用大量时间
    - GPU 计算时间很短，但总时间很长
    """
    print("=== 不好的做法：频繁小数据传输 ===")
    print("问题：每次迭代都传输数据，小传输的固定开销占比高\n")
    
    with nvtx.annotate("频繁数据传输示例", color=get_color("red")):
        for i in range(iterations):
            # CPU 端准备数据
            with nvtx.annotate(f"迭代 {i}: CPU准备数据", color=get_color("yellow")):
                cpu_data = np.random.rand(size).astype(np.float32)
            
            # ❌ 问题点 1: 频繁的小数据传输
            # 每次迭代都传输，固定开销占比高
            # 小传输（如 1KB）的效率通常 < 1%
            # 因为：固定开销（启动、同步）>> 实际传输时间
            with nvtx.annotate(f"迭代 {i}: CPU->GPU传输", color=get_color("red")):
                if USE_TORCH:
                    gpu_data = torch.from_numpy(cpu_data).to(DEVICE)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    gpu_data = cp.asarray(cpu_data)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.001)  # 模拟传输
            
            # GPU 计算（矩阵乘法）
            with nvtx.annotate(f"迭代 {i}: GPU计算", color=get_color("blue")):
                if USE_TORCH:
                    result = torch.matmul(gpu_data.unsqueeze(0), gpu_data.unsqueeze(0).t())
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    result = cp.matmul(gpu_data.reshape(1, -1), gpu_data.reshape(-1, 1))
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.0005)  # 模拟计算
            
            # ❌ 问题点 2: 立即回传结果
            # 如果不需要立即使用结果，这是浪费
            # 阻塞 GPU 继续计算
            with nvtx.annotate(f"迭代 {i}: GPU->CPU传输", color=get_color("red")):
                if USE_TORCH:
                    cpu_result = result.cpu().numpy()
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    cpu_result = cp.asnumpy(result)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.001)  # 模拟回传

def good_practice_batch_transfer(size=1000, iterations=100):
    """
    好的做法：批量传输数据
    
    优化分析：
    ----------
    这个函数展示了正确的数据传输方式：
    
    1. 批量准备数据（优化点 1）
       - 一次性准备所有数据
       - 避免循环中的数据准备开销
    
    2. 批量传输（优化点 2）
       - 一次性传输所有数据
       - 大传输的效率高（固定开销占比低）
       - 充分利用 PCIe 带宽
    
    3. 批量处理（优化点 3）
       - GPU 可以连续计算，不需要等待传输
       - 提高 GPU 利用率
    
    性能计算：
    - 批量传输（100KB）: 固定开销 ~0.5ms，有效传输 ~0.1ms
    - 效率 = 0.1 / 0.6 ≈ 16.7%
    - 1 次传输 = 0.6ms（vs 100 次小传输的 50.1ms）
    - 性能提升：50.1 / 0.6 ≈ 83.5x
    
    在 nsys 时间线中你会看到：
    - 只有一个"批量传输"标记，持续时间短
    - 大部分时间用于 GPU 计算
    - 总执行时间大幅减少
    """
    print("=== 好的做法：批量数据传输 ===")
    print("优化：批量准备和传输数据，减少传输次数，提高效率\n")
    
    with nvtx.annotate("批量传输示例", color=get_color("green")):
        # ✅ 优化点 1: 批量准备所有数据
        # 在循环外一次性准备，避免循环开销
        with nvtx.annotate("批量准备数据", color=get_color("yellow")):
            all_data = np.random.rand(iterations, size).astype(np.float32)
        
        # ✅ 优化点 2: 一次性传输所有数据
        # 大传输的效率高：固定开销相同，但有效传输时间占比高
        # 例如：100KB 传输，固定开销 0.5ms，传输时间 0.1ms
        # 效率 = 0.1 / 0.6 ≈ 16.7%（vs 小传输的 0.2%）
        with nvtx.annotate("批量CPU->GPU传输", color=get_color("purple")):
            if USE_TORCH:
                gpu_all_data = torch.from_numpy(all_data).to(DEVICE)
                torch.cuda.synchronize()
            elif USE_CUPY:
                gpu_all_data = cp.asarray(all_data)
                cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.01)  # 模拟传输
        
        # ✅ 优化点 3: 批量处理
        # GPU 可以连续计算，不需要等待传输
        # 提高 GPU 利用率和吞吐量
        with nvtx.annotate("批量GPU计算", color=get_color("blue")):
            if USE_TORCH:
                for i in range(iterations):
                    with nvtx.annotate(f"处理批次{i}", color=get_color("green")):
                        result = torch.matmul(gpu_all_data[i:i+1], gpu_all_data[i:i+1].t())
                torch.cuda.synchronize()
            elif USE_CUPY:
                for i in range(iterations):
                    with nvtx.annotate(f"处理批次{i}", color=get_color("green")):
                        result = cp.matmul(gpu_all_data[i:i+1], gpu_all_data[i:i+1].T)
                cp.cuda.Stream.null.synchronize()
            else:
                for i in range(iterations):
                    with nvtx.annotate(f"处理批次{i}", color=get_color("green")):
                        time.sleep(0.0005)
        
        # 一次性传回结果（如果需要）
        # 如果不需要结果，可以完全避免回传
        with nvtx.annotate("批量GPU->CPU传输", color=get_color("purple")):
            if USE_TORCH:
                cpu_results = gpu_all_data.cpu().numpy()
                torch.cuda.synchronize()
            elif USE_CUPY:
                cpu_results = cp.asnumpy(gpu_all_data)
                cp.cuda.Stream.null.synchronize()
            else:
                time.sleep(0.01)

if __name__ == "__main__":
    print("CPU-GPU 数据传输性能分析示例\n")
    
    # 不好的做法
    start = time.time()
    bad_practice_small_transfers(size=1000, iterations=50)
    bad_time = time.time() - start
    print(f"频繁传输耗时: {bad_time:.4f} 秒\n")
    
    # 好的做法
    start = time.time()
    good_practice_batch_transfer(size=1000, iterations=50)
    good_time = time.time() - start
    print(f"批量传输耗时: {good_time:.4f} 秒\n")
    
    print(f"性能提升: {bad_time/good_time:.2f}x")
    print("\n使用 nsys profile 查看数据传输时间线：")
    print("nsys profile --trace=cuda,nvtx,osrt --output=example2_data_transfer.nsys-rep python example2_data_transfer.py")

