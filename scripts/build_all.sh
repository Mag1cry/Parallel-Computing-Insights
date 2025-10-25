#!/bin/bash

# Parallel-Computing-Insights 项目构建脚本
# 自动编译所有CUDA实验程序

echo "=== Parallel-Computing-Insights 项目构建 ==="
echo "编译所有CUDA实验程序"
echo ""

# 检查CUDA环境
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA编译器(nvcc)未找到"
    echo "请先安装CUDA Toolkit"
    exit 1
fi

# 显示CUDA版本信息
echo "📋 环境信息："
nvcc --version | head -n 1
echo ""

# 创建输出目录
mkdir -p ../bin

# 实验1：基础GPU加法
echo "🔧 编译实验1：基础GPU加法 (simple_add)"
nvcc -o ../bin/simple_add ../001-CUDA-Integration/simple_add.cu
if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
else
    echo "❌ 编译失败"
fi

# 实验2：数组并行加法
echo ""
echo "🔧 编译实验2：数组并行加法 (array_add)"
nvcc -o ../bin/array_add ../001-CUDA-Integration/array_add.cu
if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
else
    echo "❌ 编译失败"
fi

# 实验3：数值积分（核心实验）
echo ""
echo "🔧 编译实验3：GPU数值积分 (integration_fixed)"
nvcc -o ../bin/integration_fixed ../001-CUDA-Integration/integration_fixed.cu
if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
else
    echo "❌ 编译失败"
fi

echo ""
echo "=== 构建完成 ==="
echo "生成的可执行文件位于 bin/ 目录："
ls -la ../bin/
echo ""
echo "🚀 运行示例："
echo "  ./bin/simple_add     # 基础GPU加法验证"
echo "  ./bin/array_add      # 数组并行处理"
echo "  ./bin/integration_fixed  # GPU数值积分（核心实验）"
echo ""
echo "📚 详细实验说明请参考 README.md"