#include <bits/stdc++.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// 每个线程块中的线程数
#define THREADS_PER_BLOCK 256

/**
 * reduce2: 消除 bank conflict 的版本
 * 优化：
 * 1. 使用反向循环（从 blockDim.x/2 开始，每次除以 2）
 * 2. 访问模式 shared_data[thread_idx] 和 shared_data[thread_idx + stride] 避免了 bank conflict
 * 3. 所有活跃线程连续访问共享内存，提高内存带宽利用率
 * 
 * 问题：
 * 1. 仍有分支发散：if (thread_idx < stride) 导致部分线程不执行
 * 2. 线程利用率：每次迭代后，一半的线程变为空闲
 */
__global__ void reduce2(float *device_input, float *device_output){
    // 共享内存数组，用于存储每个线程块内的部分和
    __shared__ float shared_data[THREADS_PER_BLOCK];

    // 每个线程从全局内存加载一个元素到共享内存
    unsigned int thread_idx = threadIdx.x;  // 线程在块内的索引
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程的全局索引
    shared_data[thread_idx] = device_input[global_idx];
    __syncthreads();  // 同步所有线程，确保所有数据加载完成

    // 在共享内存中进行归约操作
    // 优化：使用反向循环，从中间开始，逐步缩小范围
    // 这种访问模式避免了 bank conflict，因为相邻线程访问相邻的内存位置
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // 只有前 stride 个线程参与归约
        // 每个线程将 shared_data[thread_idx] 和 shared_data[thread_idx + stride] 相加
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

    // 计算线程块数量
    int num_blocks = num_elements / THREADS_PER_BLOCK;
    
    // 分配输出数组（每个线程块产生一个结果）
    float *host_output_data = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));
    float *device_output_data;
    cudaMalloc((void **)&device_output_data, (num_elements / THREADS_PER_BLOCK) * sizeof(float));
    
    // CPU 计算的参考结果，用于验证
    float *reference_result = (float *)malloc((num_elements / THREADS_PER_BLOCK) * sizeof(float));

    // 初始化输入数据为 1
    for(int element_idx = 0; element_idx < num_elements; element_idx++){
        host_input_data[element_idx] = 1;
    }

    // CPU 端计算参考结果（每个线程块对应的数据段的和）
    for(int block_idx = 0; block_idx < num_blocks; block_idx++){
        float partial_sum = 0;
        for(int thread_idx = 0; thread_idx < THREADS_PER_BLOCK; thread_idx++){
            partial_sum += host_input_data[block_idx * THREADS_PER_BLOCK + thread_idx];
        }
        reference_result[block_idx] = partial_sum;
    }

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(device_input_data, host_input_data, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // 配置网格和线程块维度
    dim3 grid_dim(num_elements / THREADS_PER_BLOCK, 1);  // 网格维度
    dim3 block_dim(THREADS_PER_BLOCK, 1);    // 线程块维度

    // 启动 GPU 内核
    reduce2<<<grid_dim, block_dim>>>(device_input_data, device_output_data);

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
