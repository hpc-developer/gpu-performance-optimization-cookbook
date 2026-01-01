#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

/**
 * warp_reduce: Warp 内归约函数
 * 优化：
 * 1. 展开最后一个 warp（32 个线程）的归约操作
 * 2. 使用 volatile 关键字防止编译器优化
 * 3. 不需要 __syncthreads()，因为 warp 内的线程是隐式同步的
 * 4. 完全展开循环，消除循环开销
 * 
 * 注意：warp 内的线程是锁步执行的，所以不需要显式同步
 */
__device__ void warp_reduce(volatile float* cache, unsigned int thread_idx){
    // Warp 大小是 32，所以最后一次归约需要 5 步（log2(32) = 5）
    // 完全展开循环，提高性能
    cache[thread_idx] += cache[thread_idx + 32];  // 步长 32
    //__syncthreads();  // 不需要，warp 内隐式同步
    cache[thread_idx] += cache[thread_idx + 16];  // 步长 16
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 8];   // 步长 8
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 4];   // 步长 4
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 2];   // 步长 2
    //__syncthreads();
    cache[thread_idx] += cache[thread_idx + 1];   // 步长 1
    //__syncthreads();
}

/**
 * reduce4: 展开最后一个 warp 的版本
 * 优化：
 * 1. 在加载时进行加法（reduce3 的优化）
 * 2. 展开最后一个 warp 的归约操作
 * 3. 减少了同步开销：最后一个 warp 不需要 __syncthreads()
 * 4. 提高了最后阶段的执行效率
 */
__global__ void reduce4(float *device_input, float *device_output){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float shared_data[THREADS_PER_BLOCK];

    // 每个线程加载两个元素并在加载时立即相加
    unsigned int thread_idx = threadIdx.x;  // 线程在块内的索引
    unsigned int global_idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    shared_data[thread_idx] = device_input[global_idx] + device_input[global_idx + blockDim.x];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 优化：只归约到 32 个元素（一个 warp 的大小）
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();  // 每次迭代后同步
    }

    // 优化：使用展开的 warp 归约处理最后 32 个元素
    // 这避免了循环开销和额外的同步操作
    if (thread_idx < 32) warp_reduce(shared_data, thread_idx);
    
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

    // 每个线程块处理 2*THREADS_PER_BLOCK 个元素（因为每个线程加载两个元素）
    int elements_per_block = 2 * THREADS_PER_BLOCK;
    int num_blocks = num_elements / elements_per_block;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *host_output_data = (float *)malloc(num_blocks * sizeof(float));
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, num_blocks * sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *reference_result = (float *)malloc(num_blocks * sizeof(float));

    // 初始化输入数据为 1
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        host_input_data[element_idx] = 1;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int block_idx = 0; block_idx < num_blocks; block_idx++){
        float partial_sum = 0;
        for(int element_idx = 0; element_idx < elements_per_block; element_idx++){
            partial_sum += host_input_data[block_idx * elements_per_block + element_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 grid_dim(num_blocks, 1);  // 网格维度（线程块数量减少了一半）
    dim3 block_dim(THREADS_PER_BLOCK, 1);  // 线程块维度

    // 启动 GPU 内核
    reduce4<<<grid_dim, block_dim>>>(device_input_data, device_output_data);

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
