#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

/**
 * reduce3: 在加载时进行加法的版本
 * 优化：
 * 1. 每个线程加载两个元素并在加载时立即相加
 * 2. 减少了全局内存访问次数（从 num_elements 次减少到 num_elements/2 次）
 * 3. 提高了内存带宽利用率
 * 4. 需要调整网格大小：grid_dim = num_elements/(2*THREADS_PER_BLOCK)
 * 
 * 注意：这种优化减少了线程块数量，从而减少了最终的归约步骤
 */
__global__ void reduce3(float *device_input, float *device_output){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float shared_data[THREADS_PER_BLOCK];

    // 优化：每个线程加载两个元素并在加载时立即相加
    // 这减少了全局内存访问次数，提高了带宽利用率
    unsigned int thread_idx = threadIdx.x;  // 线程在块内的索引
    // 计算全局索引：每个线程块处理 2*blockDim.x 个元素
    unsigned int global_idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    // 加载两个元素并立即相加，减少后续的归约步骤
    shared_data[thread_idx] = device_input[global_idx] + device_input[global_idx + blockDim.x];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 使用反向循环，避免 bank conflict
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        __syncthreads();  // 每次迭代后同步
    }

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
    reduce3<<<grid_dim, block_dim>>>(device_input_data, device_output_data);

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
