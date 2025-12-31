#!/bin/bash
# Profile script for example7_comprehensive.py
# 综合性能问题分析

set -e

SCRIPT_NAME="example7_comprehensive.py"
OUTPUT_FILE="example7_comprehensive.nsys-rep"

echo "=========================================="
echo "Profiling: $SCRIPT_NAME"
echo "=========================================="
echo ""

# 检查nsys是否安装
if ! command -v nsys &> /dev/null; then
    echo "错误: nsys 未找到，请安装 NVIDIA Nsight Systems"
    exit 1
fi

# 检查Python文件是否存在
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "错误: $SCRIPT_NAME 未找到"
    exit 1
fi

echo "执行命令:"
echo "  nsys profile --trace=cuda,nvtx --force-overwrite=true --output=$OUTPUT_FILE python3 $SCRIPT_NAME"
echo ""

# 执行profile
nsys profile --trace=cuda,nvtx --force-overwrite=true --output="$OUTPUT_FILE" python3 "$SCRIPT_NAME" || true

echo ""
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ 完成！生成文件: $OUTPUT_FILE"
    echo ""
    echo "查看结果:"
    echo "  - 使用 Nsight Systems GUI: nsys-ui $OUTPUT_FILE"
    echo "  - 查看统计信息: nsys stats $OUTPUT_FILE"
else
    echo "⚠ 警告: 文件未生成，请检查错误信息"
fi
echo ""

