#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256
// Warp 大小：CUDA 中一个 warp 包含 32 个线程
#define WARP_SIZE 32

/**
 * warp_reduce_sum: 使用 Shuffle 指令的 Warp 内归约函数
 * 优化：
 * 1. 使用 __shfl_down_sync 指令进行 warp 内数据交换
 * 2. 不需要共享内存，直接在寄存器间交换数据
 * 3. 延迟更低，带宽更高（寄存器访问比共享内存快）
 * 4. 使用 __forceinline__ 强制内联，减少函数调用开销
 * 5. 0xffffffff 是掩码，表示所有 32 个线程都参与
 * 
 * Shuffle 指令说明：
 * __shfl_down_sync(mask, var, delta) 从索引为 (lane_id + delta) 的线程获取 var 的值
 * 例如：lane_id=0 的线程获取 lane_id=16 的线程的值（当 delta=16 时）
 */
template <unsigned int block_size>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
    // 使用 Shuffle 指令进行 warp 内归约
    // 每个步骤将步长减半，直到所有值归约到 lane_id=0 的线程
    if (block_size >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (block_size >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (block_size >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);   // 0-4, 1-5, 2-6, etc.
    if (block_size >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);   // 0-2, 1-3, 4-6, 5-7, etc.
    if (block_size >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);   // 0-1, 2-3, 4-5, etc.
    return sum;
}

/**
 * reduce7: 使用 Shuffle 指令的最终优化版本
 * 优化：
 * 1. 每个线程处理多个元素（reduce6 的优化）
 * 2. 使用 Shuffle 指令进行 warp 内归约，避免共享内存访问
 * 3. 两阶段归约：
 *    a. 每个 warp 内部使用 Shuffle 指令归约
 *    b. 使用第一个 warp 归约所有 warp 的结果
 * 4. 减少了共享内存的使用和 bank conflict
 * 5. 这是目前最高效的归约实现之一
 * 
 * 性能优势：
 * - Shuffle 指令延迟低（~1 cycle）
 * - 不需要共享内存同步
 * - 寄存器访问比共享内存快
 */
template <unsigned int block_size, int elements_per_thread>
__global__ void reduce7(float *device_input, float *device_output){
    // 使用寄存器存储每个线程的部分和
    float sum = 0;

    // 每个线程加载并累加 elements_per_thread 个元素
    unsigned int thread_idx = threadIdx.x;  // 线程在块内的索引
    unsigned int global_idx = blockIdx.x * (block_size * elements_per_thread) + threadIdx.x;

    // 展开循环，每个线程处理多个元素
    #pragma unroll
    for(int iteration_idx = 0; iteration_idx < elements_per_thread; iteration_idx++){
        sum += device_input[global_idx + iteration_idx * block_size];
    }
    
    // 共享内存：存储每个 warp 的部分和（每个线程块最多有 WARP_SIZE 个 warp）
    static __shared__ float warp_level_sums[WARP_SIZE]; 
    const int lane_id = threadIdx.x % WARP_SIZE;  // 线程在 warp 内的索引（0-31）
    const int warp_id = threadIdx.x / WARP_SIZE;   // warp 在块内的索引

    // 第一阶段：使用 Shuffle 指令在每个 warp 内进行归约
    // 结果存储在每个 warp 的 lane_id=0 的线程中
    sum = warp_reduce_sum<block_size>(sum);

    // 将每个 warp 的归约结果写入共享内存
    if(lane_id == 0) warp_level_sums[warp_id] = sum;
    __syncthreads();  // 同步所有 warp，确保所有 warp 的结果都已写入
    
    // 第二阶段：使用第一个 warp 归约所有 warp 的结果
    // 只有前 blockDim.x/WARP_SIZE 个线程参与（即第一个 warp）
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_level_sums[lane_id] : 0;
    
    // 在第一个 warp 内进行最终归约
    if (warp_id == 0) sum = warp_reduce_sum<block_size / WARP_SIZE>(sum); 
    
    // 将当前线程块的归约结果写入全局内存
    if (thread_idx == 0) device_output[blockIdx.x] = sum;
}


/**
 * 检查函数：验证 GPU 计算结果是否正确
 * @param host_output_data GPU 计算结果数组
 * @param reference_result CPU 计算的参考结果数组
 * @param num_elements 数组长度
 * @return true 如果结果匹配，false 否则
 */
bool verify_result(float *host_output_data, float *reference_result, int num_elements){
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        if(host_output_data[element_idx] != reference_result[element_idx])
            return false;
    }
    return true;
}

int main(){
    // 数据大小：32MB 的浮点数数组
    const int num_elements = 32 * 1024 * 1024;
    
    // 分配主机内存
    float *host_input_data = (float *)malloc(num_elements * sizeof(float));
    
    // 分配设备内存
    float *device_input_data;
    cudaMalloc((void **)&device_input_data, num_elements * sizeof(float));

    // 固定线程块数量为 1024
    const int num_blocks = 1024;
    // 计算每个线程块需要处理的元素数量
    const int elements_per_block = num_elements / num_blocks;
    // 计算每个线程需要处理的元素数量
    const int elements_per_thread = elements_per_block / THREADS_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *host_output_data = (float *)malloc(num_blocks * sizeof(float));
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, num_blocks * sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *reference_result = (float *)malloc(num_blocks * sizeof(float));

    // 初始化输入数据（使用模运算生成测试数据）
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        host_input_data[element_idx] = element_idx % 456;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int block_idx = 0; block_idx < num_blocks; block_idx++){
        float partial_sum = 0;
        for(int element_idx = 0; element_idx < elements_per_block; element_idx++){
            if(block_idx * elements_per_block + element_idx < num_elements){
                partial_sum += host_input_data[block_idx * elements_per_block + element_idx];
            }
        }
        reference_result[block_idx] = partial_sum;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 grid_dim(num_blocks, 1);  // 网格维度
    dim3 block_dim(THREADS_PER_BLOCK, 1);  // 线程块维度

    // 多次迭代执行内核，用于性能测试
    int num_iterations = 2000;
    for(int iteration_idx = 0; iteration_idx < num_iterations; iteration_idx++){
        reduce7<THREADS_PER_BLOCK, elements_per_thread><<<grid_dim, block_dim>>>(device_input_data, device_output_data);
    }

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(host_output_data, device_output_data, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    if(verify_result(host_output_data, reference_result, num_blocks))printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for(int block_idx = 0; block_idx < num_blocks; block_idx++){
            printf("%lf ", host_output_data[block_idx]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(device_input_data);
    cudaFree(device_output_data);
}
