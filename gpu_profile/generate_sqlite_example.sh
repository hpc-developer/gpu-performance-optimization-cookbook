#!/bin/bash
# 示例：如何生成 SQLite 文件

set -e

echo "=========================================="
echo "生成包含 SQLite 的性能分析文件"
echo "=========================================="
echo ""

# 检查nsys是否安装
if ! command -v nsys &> /dev/null; then
    echo "错误: nsys 未找到，请安装 NVIDIA Nsight Systems"
    exit 1
fi

# 检查Python文件是否存在
if [ ! -f "example1_memory_allocation.py" ]; then
    echo "错误: example1_memory_allocation.py 未找到"
    exit 1
fi

# 输出文件名与Python示例文件名一致
OUTPUT_FILE="example1_memory_allocation.nsys-rep"

echo "运行 nsys profile 并导出 SQLite..."
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --export=sqlite \
  --force-overwrite=true \
  --output="$OUTPUT_FILE" \
  python3 example1_memory_allocation.py || true

echo ""
echo "检查生成的文件："
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ .nsys-rep 文件已生成: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE" 2>/dev/null
fi

if [ -f "${OUTPUT_FILE%.nsys-rep}.sqlite" ]; then
    echo "✓ SQLite 文件已生成: ${OUTPUT_FILE%.nsys-rep}.sqlite"
    ls -lh "${OUTPUT_FILE%.nsys-rep}.sqlite" 2>/dev/null
    echo ""
    echo "可以查看 SQLite 内容："
    echo "  sqlite3 ${OUTPUT_FILE%.nsys-rep}.sqlite '.tables'"
else
    echo "⚠ 注意: SQLite 文件未生成（可能nsys版本不支持或导出失败）"
fi
echo ""
