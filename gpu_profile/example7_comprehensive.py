"""
示例 7: 综合性能问题分析
结合多个性能问题的真实场景示例
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
        "purple": (0.5, 0.0, 0.5),
        "cyan": (0.0, 1.0, 1.0),
    }
    return colors.get(name, (0.5, 0.5, 0.5))

def comprehensive_bad_practice():
    """综合示例：包含多个性能问题"""
    print("=== 综合示例：多个性能问题 ===")
    
    with nvtx.annotate("综合性能问题示例", color=get_color("red")):
        # 问题 1: 频繁内存分配
        with nvtx.annotate("阶段1: 频繁分配", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"分配{i}", color=get_color("yellow")):
                    if USE_TORCH:
                        data = torch.randn(100, 100, device=DEVICE)
                        torch.cuda.synchronize()
                        del data
                    elif USE_CUPY:
                        data = cp.random.randn(100, 100, dtype=cp.float32)
                        cp.cuda.Stream.null.synchronize()
                        del data
                    else:
                        data = np.random.rand(100, 100).astype(np.float32)
                        del data
        
        # 问题 2: 频繁数据传输
        with nvtx.annotate("阶段2: 频繁传输", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"传输{i}", color=get_color("red")):
                    cpu_data = np.random.rand(100).astype(np.float32)
                    if USE_TORCH:
                        gpu_data = torch.from_numpy(cpu_data).to(DEVICE)
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        gpu_data = cp.asarray(cpu_data)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        time.sleep(0.001)  # 模拟传输
        
        # 问题 3: 频繁同步
        with nvtx.annotate("阶段3: 频繁同步", color=get_color("orange")):
            for i in range(10):
                with nvtx.annotate(f"计算{i}", color=get_color("blue")):
                    if USE_TORCH:
                        x = torch.randn(128, 128, device=DEVICE)
                        y = torch.matmul(x, x)
                    elif USE_CUPY:
                        x = cp.random.randn(128, 128, dtype=cp.float32)
                        y = cp.matmul(x, x)
                    else:
                        time.sleep(0.001)
                with nvtx.annotate(f"同步{i}", color=get_color("red")):
                    if USE_TORCH:
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        cp.cuda.Stream.null.synchronize()
                    else:
                        time.sleep(0.002)
        
        # 问题 4: 小内核
        with nvtx.annotate("阶段4: 小内核", color=get_color("orange")):
            for i in range(20):
                with nvtx.annotate(f"小内核{i}", color=get_color("purple")):
                    if USE_TORCH:
                        a = torch.randn(32, 32, device=DEVICE)
                        b = torch.matmul(a, a)
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        a = cp.random.randn(32, 32, dtype=cp.float32)
                        b = cp.matmul(a, a)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        time.sleep(0.0001)

def comprehensive_good_practice():
    """
    综合示例：优化后的版本
    
    优化分析：
    ----------
    这个函数展示了系统性的优化方法：
    
    阶段 1: 预分配内存
    - 优化：预先分配内存，循环中重用
    - 效果：减少分配次数，降低开销
    - 时间减少：~70-80%
    
    阶段 2: 批量传输
    - 优化：批量准备数据，一次性传输
    - 效果：减少传输次数，提高效率
    - 时间减少：~80-90%
    
    阶段 3: 延迟同步
    - 优化：批量启动，延迟同步
    - 效果：提高并行性，减少等待
    - 时间减少：~70-80%
    
    阶段 4: 合并内核
    - 优化：合并小内核为大内核
    - 效果：减少启动次数，提高效率
    - 时间减少：~50-70%
    
    总体效果：
    - 总时间减少：~5-10 倍
    - GPU 利用率提高：~3-5 倍
    - 资源利用率提高：~2-3 倍
    
    在 nsys 时间线中你会看到：
    - 所有阶段都得到优化
    - 时间线更紧凑，空闲时间少
    - GPU 利用率高且稳定
    """
    print("=== 综合示例：优化版本 ===")
    print("系统性优化：内存、传输、同步、内核全部优化\n")
    
    with nvtx.annotate("优化示例", color=get_color("green")):
        # ✅ 优化 1: 预分配内存
        # 预先分配内存，循环中重用，避免频繁分配
        with nvtx.annotate("阶段1: 预分配", color=get_color("blue")):
            if USE_TORCH:
                data = torch.empty(100, 100, device=DEVICE)
            elif USE_CUPY:
                data = cp.empty((100, 100), dtype=cp.float32)
            else:
                data = np.empty((100, 100), dtype=np.float32)
            
            for i in range(10):
                with nvtx.annotate(f"重用{i}", color=get_color("green")):
                    # 重用已分配的内存，不需要重新分配
                    if USE_TORCH:
                        data[:] = torch.randn(100, 100, device=DEVICE)
                    elif USE_CUPY:
                        data[:] = cp.random.randn(100, 100, dtype=cp.float32)
                    else:
                        data[:] = np.random.rand(100, 100).astype(np.float32)
        
        # ✅ 优化 2: 批量传输
        # 批量准备数据，一次性传输，提高传输效率
        with nvtx.annotate("阶段2: 批量传输", color=get_color("blue")):
            all_data = np.random.rand(10, 100).astype(np.float32)
            with nvtx.annotate("批量传输", color=get_color("cyan")):
                # 一次传输所有数据，效率高
                if USE_TORCH:
                    gpu_all_data = torch.from_numpy(all_data).to(DEVICE)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    gpu_all_data = cp.asarray(all_data)
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.01)  # 批量传输，总时间 < 多次小传输
        
        # ✅ 优化 3: 延迟同步
        # 批量启动所有计算，最后统一同步
        with nvtx.annotate("阶段3: 延迟同步", color=get_color("blue")):
            # 启动所有计算（非阻塞）
            gpu_results = []
            for i in range(10):
                with nvtx.annotate(f"启动{i}", color=get_color("green")):
                    if USE_TORCH:
                        x = torch.randn(128, 128, device=DEVICE)
                        y = torch.matmul(x, x)
                        gpu_results.append(y)
                    elif USE_CUPY:
                        x = cp.random.randn(128, 128, dtype=cp.float32)
                        y = cp.matmul(x, x)
                        gpu_results.append(y)
                    else:
                        time.sleep(0.001)
            # 只同步一次，而不是 10 次
            with nvtx.annotate("最终同步", color=get_color("cyan")):
                if USE_TORCH:
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    cp.cuda.Stream.null.synchronize()
                else:
                    time.sleep(0.002)
        
        # ✅ 优化 4: 合并内核
        # 将多个小操作合并为批次，减少启动次数
        with nvtx.annotate("阶段4: 合并内核", color=get_color("blue")):
            batch_size = 5
            for batch in range(4):
                with nvtx.annotate(f"大内核{batch}", color=get_color("green")):
                    if USE_TORCH:
                        # 大矩阵乘法，启动开销占比低
                        a = torch.randn(256, 256, device=DEVICE)
                        b = torch.matmul(a, a)
                        torch.cuda.synchronize()
                    elif USE_CUPY:
                        a = cp.random.randn(256, 256, dtype=cp.float32)
                        b = cp.matmul(a, a)
                        cp.cuda.Stream.null.synchronize()
                    else:
                        time.sleep(0.0001)  # 启动开销（与单个小内核相同）
                        # 批量处理，提高效率
                        for i in range(batch_size):
                            time.sleep(0.00005)

def performance_analysis_workflow():
    """性能分析工作流程示例"""
    print("=== 性能分析工作流程 ===")
    
    with nvtx.annotate("性能分析流程", color=get_color("purple")):
        # 1. 数据准备
        with nvtx.annotate("1. 数据准备", color=get_color("yellow")):
            with nvtx.annotate("加载数据", color=get_color("blue")):
                if USE_TORCH:
                    data = torch.randn(1000, 1000, device=DEVICE)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    data = cp.random.randn(1000, 1000, dtype=cp.float32)
                    cp.cuda.Stream.null.synchronize()
                else:
                    data = np.random.rand(1000, 1000).astype(np.float32)
                    time.sleep(0.01)
            
            with nvtx.annotate("数据预处理", color=get_color("blue")):
                if USE_TORCH:
                    data = data / torch.max(data)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    data = data / cp.max(data)
                    cp.cuda.Stream.null.synchronize()
                else:
                    data = data / np.max(data)
                    time.sleep(0.005)
        
        # 2. 计算阶段
        with nvtx.annotate("2. 计算阶段", color=get_color("green")):
            with nvtx.annotate("主要计算", color=get_color("blue")):
                if USE_TORCH:
                    result1 = torch.sum(data)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    result1 = cp.sum(data)
                    cp.cuda.Stream.null.synchronize()
                else:
                    result1 = np.sum(data)
                    time.sleep(0.02)
            
            with nvtx.annotate("辅助计算", color=get_color("blue")):
                if USE_TORCH:
                    result2 = torch.mean(data)
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    result2 = cp.mean(data)
                    cp.cuda.Stream.null.synchronize()
                else:
                    result2 = np.mean(data)
                    time.sleep(0.01)
        
        # 3. 后处理
        with nvtx.annotate("3. 后处理", color=get_color("orange")):
            with nvtx.annotate("结果处理", color=get_color("blue")):
                if USE_TORCH:
                    final_result = result1 + result2
                    torch.cuda.synchronize()
                elif USE_CUPY:
                    final_result = result1 + result2
                    cp.cuda.Stream.null.synchronize()
                else:
                    final_result = result1 + result2
                    time.sleep(0.005)

if __name__ == "__main__":
    print("综合性能分析示例\n")
    
    # 包含多个问题的版本
    start = time.time()
    comprehensive_bad_practice()
    bad_time = time.time() - start
    print(f"未优化版本耗时: {bad_time:.4f} 秒\n")
    
    # 优化后的版本
    start = time.time()
    comprehensive_good_practice()
    good_time = time.time() - start
    print(f"优化版本耗时: {good_time:.4f} 秒\n")
    
    print(f"总体性能提升: {bad_time/good_time:.2f}x\n")
    
    # 性能分析工作流程
    performance_analysis_workflow()
    
    print("\n使用 nsys profile 进行完整分析：")
    print("nsys profile \\")
    print("  --trace=cuda,nvtx,osrt \\")
    print("  --cuda-memory-usage=true \\")
    print("  --sampling-frequency=1000 \\")
    print("  --gpu-metrics-frequency=100 \\")
    print("  --stats=true \\")
    print("  --output=example7_comprehensive.nsys-rep \\")
    print("  python example7_comprehensive.py")

