#include <stdio.h>
#include <cuda_runtime.h>

// 实验：用GPU并行计算积分 ∫x² dx 从0到1（解析解是1/3）
__global__ void integrate_square(float* result, int num_segments) {
    __shared__ float shared_sum[256]; // 共享内存，线程块内可见
    
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    float dx = 1.0f / num_segments;
    float local_sum = 0.0f;
    
    // 每个线程处理一部分区间
    for(int i = tid; i < num_segments; i += total_threads) {
        float x = (i + 0.5f) * dx;
        local_sum += x * x * dx;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads(); // 等待所有线程完成写入
    
    // 并行归约：树形结构求和
    for(int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // 只有线程0将最终结果写入全局内存
    if(tid == 0) {
        *result = shared_sum[0];
    }
}

int main() {
    const int num_threads = 256;  // 增加到256个线程
    const int num_segments = 1000000; // 增加到100万份
    
    float host_result = 0;
    float* dev_result;
    cudaMalloc(&dev_result, sizeof(float));
    
    integrate_square<<<1, num_threads>>>(dev_result, num_segments);
    
    cudaMemcpy(&host_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("修复后数值积分结果: %f\n", host_result);
    printf("解析解(1/3): 0.333333\n");
    printf("误差: %f\n", fabs(host_result - 0.333333f));
    
    cudaFree(dev_result);
    return 0;
}