#include <stdio.h>
#include <cuda_runtime.h>

// 实验2：并行扩展 - 数组并行处理
// 体验真正的"子弹齐射"并行模式

__global__ void array_add_kernel(int* a, int* b, int* result, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        result[tid] = a[tid] + b[tid];
    }
}

int main() {
    printf("=== 实验2：数组并行加法 ===\n");
    printf("体验真正的GPU并行计算能力\n\n");
    
    const int SIZE = 1000;
    const int BLOCK_SIZE = 256;
    
    // 主机端数组
    int host_a[SIZE], host_b[SIZE], host_result[SIZE];
    
    // 初始化数据
    for (int i = 0; i < SIZE; i++) {
        host_a[i] = i;
        host_b[i] = i * 2;
    }
    
    // 设备端数组指针
    int *dev_a, *dev_b, *dev_result;
    
    // GPU内存分配
    cudaMalloc(&dev_a, SIZE * sizeof(int));
    cudaMalloc(&dev_b, SIZE * sizeof(int));
    cudaMalloc(&dev_result, SIZE * sizeof(int));
    
    // 数据复制到GPU
    cudaMemcpy(dev_a, host_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    
    // 计算网格和块大小
    int num_blocks = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("并行配置: %d个线程块 × %d个线程/块 = %d个并行线程\n", 
           num_blocks, BLOCK_SIZE, num_blocks * BLOCK_SIZE);
    printf("处理数组大小: %d个元素\n\n", SIZE);
    
    // 启动并行内核
    array_add_kernel<<<num_blocks, BLOCK_SIZE>>>(dev_a, dev_b, dev_result, SIZE);
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    // 结果传回CPU
    cudaMemcpy(host_result, dev_result, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool correct = true;
    for (int i = 0; i < SIZE; i++) {
        if (host_result[i] != host_a[i] + host_b[i]) {
            correct = false;
            break;
        }
    }
    
    printf("验证结果: %s\n", correct ? "✓ 所有元素计算正确" : "✗ 存在计算错误");
    
    // 显示部分结果
    printf("\n前10个元素计算结果:\n");
    for (int i = 0; i < 10; i++) {
        printf("a[%d]=%d + b[%d]=%d = result[%d]=%d\n", 
               i, host_a[i], i, host_b[i], i, host_result[i]);
    }
    
    // 清理GPU内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    
    printf("\n=== 实验总结 ===\n");
    printf("1. 体验了真正的"子弹齐射"并行模式\n");
    printf("2. 掌握了线程块和网格的组织方式\n");
    printf("3. 理解了大规模数据并行处理的威力\n");
    printf("4. 学会了GPU内存的批量分配和传输\n");
    
    return 0;
}