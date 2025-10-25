#include <stdio.h>
#include <cuda_runtime.h>

// 实验1：基础验证 - 简单GPU加法
// 理解CPU/GPU数据传输机制

__global__ void add_kernel(int a, int b, int* result) {
    *result = a + b;
}

int main() {
    printf("=== 实验1：基础GPU加法验证 ===\n");
    printf("验证CPU/GPU数据传输和基本并行计算\n\n");
    
    int a = 123, b = 456;
    int host_result = 0;
    int* dev_result;
    
    // GPU内存分配
    cudaMalloc(&dev_result, sizeof(int));
    
    // 启动GPU内核
    add_kernel<<<1, 1>>>(a, b, dev_result);
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    // 结果传回CPU
    cudaMemcpy(&host_result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("CPU计算: %d + %d = %d\n", a, b, a + b);
    printf("GPU计算: %d + %d = %d\n", a, b, host_result);
    printf("验证结果: %s\n", (a + b == host_result) ? "✓ 正确" : "✗ 错误");
    
    // 清理GPU内存
    cudaFree(dev_result);
    
    printf("\n=== 实验总结 ===\n");
    printf("1. 掌握了GPU内存分配(cudaMalloc)\n");
    printf("2. 理解了内核启动语法(<<< >>>)\n");
    printf("3. 学会了数据同步(cudaDeviceSynchronize)\n");
    printf("4. 熟悉了数据传输(cudaMemcpy)\n");
    
    return 0;
}