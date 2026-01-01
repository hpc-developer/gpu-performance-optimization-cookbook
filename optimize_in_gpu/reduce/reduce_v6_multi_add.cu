#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

/**
 * warp_reduce: 模板化的 Warp 内归约函数
 * 与 reduce5 中的实现相同，用于 warp 内的最后归约步骤
 */
template <unsigned int block_size>
__device__ void warp_reduce(volatile float* cache, unsigned int thread_idx){
    if (block_size >= 64) cache[thread_idx] += cache[thread_idx + 32];
    if (block_size >= 32) cache[thread_idx] += cache[thread_idx + 16];
    if (block_size >= 16) cache[thread_idx] += cache[thread_idx + 8];
    if (block_size >= 8) cache[thread_idx] += cache[thread_idx + 4];
    if (block_size >= 4) cache[thread_idx] += cache[thread_idx + 2];
    if (block_size >= 2) cache[thread_idx] += cache[thread_idx + 1];
}

/**
 * reduce6: 每个线程处理多个元素的版本
 * 优化：
 * 1. 每个线程处理 elements_per_thread 个元素，减少线程块数量
 * 2. 使用 #pragma unroll 展开循环，提高性能
 * 3. 减少了全局内存访问的延迟影响
 * 4. 提高了 GPU 的占用率（occupancy）
 * 5. 结合了 reduce5 的所有优化（完全展开、warp 归约等）
 * 
 * 这种优化特别适合处理大量数据的情况
 */
template <unsigned int block_size, int elements_per_thread>
__global__ void reduce6(float *device_input, float *device_output){
    // 共享内存数组，大小等于线程块大小
    __shared__ float shared_data[block_size];

    // 每个线程处理 elements_per_thread 个元素
    unsigned int thread_idx = threadIdx.x;  // 线程在块内的索引
    // 计算起始全局索引：每个线程块处理 block_size * elements_per_thread 个元素
    unsigned int global_idx = blockIdx.x * (block_size * elements_per_thread) + threadIdx.x;

    // 初始化共享内存中的累加器
    shared_data[thread_idx] = 0;

    // 优化：使用 #pragma unroll 展开循环
    // 每个线程加载并累加 elements_per_thread 个元素
    // 这减少了线程块数量，提高了内存带宽利用率
    #pragma unroll
    for(int iteration_idx = 0; iteration_idx < elements_per_thread; iteration_idx++){
        // 使用跨步访问模式：相邻线程访问间隔 block_size 的元素
        // 这有助于合并内存访问，提高带宽利用率
        shared_data[thread_idx] += device_input[global_idx + iteration_idx * block_size];
    }
    
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作（与 reduce5 相同）
    // 完全展开归约循环
    if (block_size >= 512) {
        if (thread_idx < 256) { 
            shared_data[thread_idx] += shared_data[thread_idx + 256]; 
        } 
        __syncthreads(); 
    }
    if (block_size >= 256) {
        if (thread_idx < 128) { 
            shared_data[thread_idx] += shared_data[thread_idx + 128]; 
        } 
        __syncthreads(); 
    }
    if (block_size >= 128) {
        if (thread_idx < 64) { 
            shared_data[thread_idx] += shared_data[thread_idx + 64]; 
        } 
        __syncthreads(); 
    }
    // 最后 32 个元素使用展开的 warp 归约
    if (thread_idx < 32) warp_reduce<block_size>(shared_data, thread_idx);
    
    // 将当前线程块的归约结果写入全局内存
    if (thread_idx == 0) device_output[blockIdx.x] = shared_data[0];
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
    
    // 启动 GPU 内核（使用模板参数指定线程块大小和每个线程处理的元素数）
    reduce6<THREADS_PER_BLOCK, elements_per_thread><<<grid_dim, block_dim>>>(device_input_data, device_output_data);

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
