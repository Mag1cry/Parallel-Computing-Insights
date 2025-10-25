# Parallel-Computing-Insights

从第一性原理探索GPU并行计算的核心范式

## 项目概述

本项目通过一系列CUDA实验，验证"分治思想、GPU并行、数学积分"共享相同的计算范式。从简单的并行加法到复杂的数值积分，逐步揭示并行计算的本质规律。

## 核心洞察

- **计算范式统一性**：分治策略、GPU并行架构、数值积分方法在计算范式上高度统一
- **并行协调挑战**：并行计算的真正挑战不仅是同时计算，更是线程间的协调与同步
- **共享内存机制**：GPU共享内存相当于线程的"作战会议室"，是实现高效并行的关键

## 实验历程

### 1. 基础验证 - 简单加法

- 理解CPU/GPU数据传输机制
- 建立基本的CUDA编程模型认知

### 2. 并行扩展 - 数组处理  

- 体验真正的"子弹齐射"并行模式
- 掌握线程块和网格的组织方式

### 3. 概念融合 - 数值积分

- 将数学积分问题映射到硬件并行架构
- 验证分治思想在GPU计算中的自然体现

### 4. 调试突破 - 精度优化

- 从误差0.322915到精确解0.333333
- 深入理解并行协调的本质

## 项目结构

```path
Parallel-Computing-Insights/
├── 001-CUDA-Integration/     # CUDA实验代码
│   ├── simple_add.cu         # 简单加法实验
│   ├── array_add.cu          # 数组并行版本  
│   ├── integration_fixed.cu  # 修复后的积分程序
│   └── README.md             # 实验记录和洞察
├── Concepts/                 # 理论文档
│   └── divide-conquer-gpu.md # 分治思想与GPU并行
├── scripts/                  # 工具脚本
│   └── compile_wsl.sh        # WSL编译脚本
└── docs/                     # 文档资料
    └── experiment-notes.md   # 实验过程记录
```

## 快速开始

### 环境要求

- CUDA Toolkit 11.0+
- NVIDIA GPU with Compute Capability 3.0+
- Linux/WSL环境（推荐）

### 编译运行

```bash
# 使用提供的编译脚本
./scripts/compile_wsl.sh

# 或手动编译
nvcc -o bin/integration_test 001-CUDA-Integration/integration_fixed.cu
./bin/integration_test
```

## 关键收获

1. **技术层面**：掌握了CUDA编程的核心概念和调试技巧
2. **理论层面**：验证了不同计算范式之间的内在联系
3. **思维层面**：培养了从第一性原理出发的探索方法

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 贡献

欢迎通过Issue和Pull Request参与项目改进和扩展。
