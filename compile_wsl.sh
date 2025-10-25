#!/bin/bash

# 将Windows文件复制到WSL中
cp /mnt/c/003Codes/cppCodes/001CUDA/CUDAtest.cu /tmp/CUDAtest.cu

# 设置完整的CUDA环境变量
# export CUDA_HOME=/usr/local/cuda-13.0
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# 检查是否安装了nvcc
if [ -f /usr/local/cuda-13.0/bin/nvcc ]; then
    echo "✅ 检测到CUDA Toolkit已安装 (nvcc 13.0.88)"
    echo "🔄 使用nvcc编译CUDA代码..."
    
    # 显示环境信息
    echo "CUDA路径: $CUDA_HOME"
    
    # 使用真正的nvcc编译CUDA代码，显示详细错误信息
    /usr/local/cuda-13.0/bin/nvcc -o /tmp/CUDAtest /tmp/CUDAtest.cu \
        -L/usr/local/cuda-13.0/lib64 \
        -lcudart \
        -I/usr/local/cuda-13.0/include
     
    if [ $? -eq 0 ]; then
        echo "✅ CUDA编译成功！运行程序："
        /tmp/CUDAtest
    else
        echo "❌ CUDA编译失败"
        echo "尝试使用更简单的编译命令..."
        
        # 尝试使用更简单的编译命令
        /usr/local/cuda-13.0/bin/nvcc -o /tmp/CUDAtest /tmp/CUDAtest.cu
        
        if [ $? -eq 0 ]; then
            echo "✅ 简化编译成功！运行程序："
            /tmp/CUDAtest
        else
            echo "❌ 编译仍然失败"
            echo "请检查WSL CUDA环境配置"
            exit 1
        fi
    fi
else
    echo "❌ CUDA Toolkit未安装"
    echo "请先安装CUDA Toolkit：sudo apt install cuda-toolkit-13-0"
    exit 1
fi