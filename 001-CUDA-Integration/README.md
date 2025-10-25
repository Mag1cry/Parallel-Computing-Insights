# 001-CUDA-Integration

CUDA并行计算实验系列 - 从基础到高级的完整学习路径

## 实验概述

本目录包含三个循序渐进的CUDA实验，通过具体的编码实践，深入理解GPU并行计算的核心概念。

## 实验文件说明

### 1. simple_add.cu - 基础GPU加法

**目标**：验证CPU/GPU数据传输机制，建立基本的CUDA编程模型认知

**关键特性**：

- 最简单的GPU内核实现
- 单线程计算模式
- 基础的内存管理操作

**学习重点**：

- `cudaMalloc` - GPU内存分配
- `<<< >>>` - 内核启动语法
- `cudaDeviceSynchronize` - 设备同步
- `cudaMemcpy` - 数据传输

### 2. array_add.cu - 数组并行加法

**目标**：体验真正的"子弹齐射"并行模式，掌握线程块和网格的组织方式

**关键特性**：

- 大规模数据并行处理
- 线程索引计算
- 网格和块的组织

**技术参数**：

- 数组大小：1000个元素
- 线程块大小：256个线程
- 总并行线程：1024个

**学习重点**：

- 线程ID计算：`threadIdx.x + blockIdx.x * blockDim.x`
- 边界检查：`if (tid < size)`
- 批量内存操作

### 3. integration_fixed.cu - GPU数值积分（核心实验）

**目标**：将数学积分问题映射到硬件并行架构，验证分治思想的自然体现

**数学问题**：计算∫x²dx从0到1，解析解为1/3

**关键技术**：

- 共享内存使用：`__shared__ float shared_sum[256]`
- 线程同步：`__syncthreads()`
- 并行归约：树形结构求和

**精度成果**：

- 计算结果：0.333333
- 解析解：0.333333
- 误差：接近机器精度

## 实验运行指南

### 编译所有实验

```bash
# Linux/WSL
./scripts/build_all.sh

# Windows
scripts\build_windows.bat
```

### 单独编译

```bash
# 基础加法
nvcc -o ../bin/simple_add simple_add.cu

# 数组加法  
nvcc -o ../bin/array_add array_add.cu

# 数值积分
nvcc -o ../bin/integration_fixed integration_fixed.cu
```

### 运行实验

```bash
# 实验1：基础验证
./bin/simple_add

# 实验2：并行扩展
./bin/array_add

# 实验3：概念融合
./bin/integration_fixed
```

## 实验设计理念

### 循序渐进的学习路径

1. **从简单到复杂**：单线程 → 多线程 → 复杂算法
2. **从具体到抽象**：具体操作 → 通用模式 → 理论认知
3. **从技术到思维**：编码技能 → 调试能力 → 系统思考

### 核心教育目标

- **技术掌握**：CUDA编程的核心概念和技巧
- **思维训练**：从第一性原理出发的探索方法
- **范式认知**：理解不同计算领域的内在统一性

## 实验关联性

三个实验不是孤立的，而是构成了完整的学习闭环：

1. **simple_add** 建立了基础认知
2. **array_add** 扩展了并行视野  
3. **integration_fixed** 实现了概念融合

通过这个系列，学习者可以：

- 亲身体验并行计算的发展历程
- 深刻理解技术演进的逻辑
- 建立系统性思考问题的能力

## 技术要点总结

### 内存管理

- 全局内存：`cudaMalloc` / `cudaFree`
- 共享内存：`__shared__`
- 数据传输：`cudaMemcpy`

### 并行组织

- 线程：`threadIdx.x`
- 线程块：`blockDim.x`, `blockIdx.x`
- 网格：自动计算或手动指定

### 同步机制

- 设备同步：`cudaDeviceSynchronize`
- 线程块同步：`__syncthreads()`

## 后续学习建议

完成本系列实验后，建议：

1. **性能优化**：尝试不同的并行策略和参数调优
2. **算法扩展**：实现其他数学问题的GPU版本
3. **应用实践**：将学到的模式应用到实际项目中
4. **理论研究**：深入探索并行计算的理论基础

> 真正的学习不在于记住答案，而在于理解问题背后的模式。
