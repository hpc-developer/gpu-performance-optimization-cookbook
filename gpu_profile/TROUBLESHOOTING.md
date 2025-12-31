# 故障排除指南

## 常见问题

### 1. 找不到 profile.sqllite 或 profile.sqlite 文件

**问题描述：**
运行 `nsys profile` 后，找不到 SQLite 数据库文件。

**原因：**
`nsys profile` 默认只生成 `.nsys-rep` 文件，不会自动生成 SQLite 文件。

**解决方法：**

#### 方法 1: 使用 --export 参数（推荐）

在运行 `nsys profile` 时添加 `--export=sqlite` 参数：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --export=sqlite \
  --output=profile.nsys-rep \
  python your_script.py
```

这会生成：
- `profile.nsys-rep` - 主报告文件
- `profile.sqlite` - SQLite 数据库文件（在同一目录）

#### 方法 2: 从现有的 .nsys-rep 文件导出

如果你已经有一个 `.nsys-rep` 文件，可以使用 `nsys export` 命令：

```bash
# 导出为 SQLite
nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep

# 或导出为多种格式
nsys export --type=sqlite,json --output=profile profile.nsys-rep
```

#### 方法 3: 使用 nsys stats 导出

```bash
# 导出统计报告（包含数据库信息）
nsys stats --report gputrace profile.nsys-rep
```

**验证：**
```bash
# 检查文件是否生成
ls -lh profile.sqlite

# 验证 SQLite 文件
sqlite3 profile.sqlite ".tables"

# 查看 NVTX 相关数据（如果使用了 NVTX 标记）
sqlite3 profile.sqlite "SELECT * FROM StringIds WHERE value LIKE '%你的标记名%';"
sqlite3 profile.sqlite "SELECT * FROM NVTX_EVENTS LIMIT 10;"
```

---

### 1.1 SQLite 文件中缺少 NVTX 信息

**问题描述：**
生成的 SQLite 文件中找不到 NVTX 标记信息。

**可能原因：**
1. 没有使用 `--trace=nvtx` 参数
2. 代码中没有添加 NVTX 标记
3. SQLite 文件中的表结构不同

**解决方法：**

#### 确保使用 --trace=nvtx

```bash
# ❌ 错误：没有跟踪 NVTX
nsys profile --trace=cuda --export=sqlite --output=profile.nsys-rep python script.py

# ✅ 正确：跟踪 NVTX
nsys profile --trace=cuda,nvtx --export=sqlite --output=profile.nsys-rep python script.py
```

#### 检查 SQLite 文件内容

```bash
# 1. 查看所有表
sqlite3 profile.sqlite ".tables"

# 2. 查看 StringIds（包含 NVTX 标记名称）
sqlite3 profile.sqlite "SELECT * FROM StringIds LIMIT 10;"

# 3. 查看事件表（包含 NVTX 事件）
sqlite3 profile.sqlite "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%EVENT%';"

# 4. 查询 NVTX 标记
sqlite3 profile.sqlite "SELECT * FROM STRING_TABLE WHERE value LIKE '%你的标记名%';"
```

#### 查看 NVTX 数据的正确方法

SQLite 文件中的 NVTX 数据存储在以下表中：
- `StringIds` - 存储字符串（包括 NVTX 标记名称）
- `NVTX_EVENTS` - 存储 NVTX 事件数据（如果使用了 `--trace=nvtx`）

```bash
# 1. 查看所有表（确认 NVTX_EVENTS 是否存在）
sqlite3 profile.sqlite ".tables" | grep -i nvtx

# 2. 查看 StringIds 表结构
sqlite3 profile.sqlite ".schema StringIds"

# 3. 查看所有 NVTX 相关的字符串
sqlite3 profile.sqlite "SELECT * FROM StringIds WHERE value LIKE '%NVTX%' OR value LIKE '%你的标记%';"

# 4. 查看 NVTX_EVENTS 表结构
sqlite3 profile.sqlite ".schema NVTX_EVENTS"

# 5. 查看 NVTX 事件数据
sqlite3 profile.sqlite "SELECT * FROM NVTX_EVENTS LIMIT 10;"

# 6. 统计 NVTX 事件数量
sqlite3 profile.sqlite "SELECT COUNT(*) FROM NVTX_EVENTS;"
```

**重要提示：**
- 如果 `NVTX_EVENTS` 表不存在或为空，说明没有使用 `--trace=nvtx` 参数
- 如果 `StringIds` 表中没有你的标记名称，说明代码中的 NVTX 标记没有被正确记录

#### 使用 GUI 工具查看

如果 SQLite 文件中确实没有 NVTX 数据，建议使用 Nsight Systems GUI：

```bash
nsys-ui profile.nsys-rep
```

GUI 工具可以更好地显示 NVTX 标记，即使 SQLite 文件中没有直接可见的数据。

#### 完整示例

```bash
# 1. 运行带 NVTX 跟踪的分析
nsys profile \
  --trace=cuda,nvtx \
  --export=sqlite \
  --output=profile.nsys-rep \
  python3 example1_memory_allocation.py

# 2. 检查 SQLite 文件
sqlite3 profile.sqlite ".tables"

# 3. 查看 NVTX 标记
sqlite3 profile.sqlite "SELECT * FROM StringIds WHERE value LIKE '%数据%' OR value LIKE '%计算%';"
sqlite3 profile.sqlite "SELECT COUNT(*) FROM NVTX_EVENTS;"

# 4. 如果找不到，使用 GUI
nsys-ui profile.nsys-rep
```

---

### 2. .nsys-rep 文件找不到

**问题描述：**
运行 `nsys profile` 后，找不到输出文件。

**可能原因：**
1. 输出路径不正确
2. 文件权限问题
3. 命令执行失败

**解决方法：**

```bash
# 1. 检查当前目录
pwd
ls -lh *.nsys-rep

# 2. 使用绝对路径
nsys profile --output=/data/code/gpu_profile/profile.nsys-rep python script.py

# 3. 检查命令是否成功执行
echo $?  # 应该返回 0

# 4. 查看详细输出
nsys profile --output=profile.nsys-rep python script.py 2>&1 | tee nsys_output.log
```

---

### 3. nsys-ui 无法打开文件

**问题描述：**
使用 `nsys-ui profile.nsys-rep` 无法打开文件。

**可能原因：**
1. 文件损坏
2. 文件格式不兼容
3. GUI 工具未安装

**解决方法：**

```bash
# 1. 检查文件是否完整
ls -lh profile.nsys-rep
file profile.nsys-rep

# 2. 验证文件
nsys stats profile.nsys-rep

# 3. 尝试使用完整路径
nsys-ui /data/code/gpu_profile/profile.nsys-rep

# 4. 检查 GUI 工具
which nsys-ui
which nsight-sys

# 5. 如果没有 GUI，使用命令行工具
nsys stats profile.nsys-rep
```

---

### 4. 导出 SQLite 文件时出错

**问题描述：**
使用 `--export=sqlite` 时出现错误。

**可能原因：**
1. 输出目录不存在
2. 文件权限问题
3. 磁盘空间不足

**解决方法：**

```bash
# 1. 确保输出目录存在
mkdir -p output_dir
nsys profile --export=sqlite --output=output_dir/profile.nsys-rep python script.py

# 2. 检查磁盘空间
df -h .

# 3. 检查文件权限
ls -ld output_dir

# 4. 使用绝对路径
nsys profile --export=sqlite --output=/data/code/gpu_profile/profile.nsys-rep python script.py
```

---

### 5. SQLite 文件位置不确定

**问题描述：**
不知道 SQLite 文件生成在哪里。

**SQLite 文件位置规则：**

1. **使用 --export=sqlite 时：**
   - SQLite 文件与 `.nsys-rep` 文件在同一目录
   - 文件名：`<output_name>.sqlite`
   - 例如：`--output=profile.nsys-rep` → `profile.sqlite`

2. **使用 nsys export 时：**
   - 使用 `--output` 指定的路径
   - 例如：`--output=profile.sqlite` → `profile.sqlite`

**查找 SQLite 文件：**

```bash
# 方法 1: 在输出目录查找
find . -name "*.sqlite" -o -name "*.sqllite"

# 方法 2: 检查 .nsys-rep 文件所在目录
ls -lh $(dirname profile.nsys-rep)/*.sqlite

# 方法 3: 使用 locate（如果已安装）
locate profile.sqlite
```

---

## 完整示例

### 生成包含 SQLite 的性能分析

```bash
cd /data/code/gpu_profile
conda activate python3.12

# 运行分析并导出 SQLite
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --export=sqlite,json \
  --output=profile.nsys-rep \
  python example1_memory_allocation.py

# 检查生成的文件
ls -lh profile.*

# 验证 SQLite 文件
sqlite3 profile.sqlite ".tables" 2>/dev/null || echo "SQLite 文件可能未生成或格式不正确"
```

### 从现有文件导出 SQLite

```bash
# 如果已有 .nsys-rep 文件
nsys export --type=sqlite --output=profile.sqlite profile.nsys-rep

# 验证
ls -lh profile.sqlite
sqlite3 profile.sqlite ".tables"
```

---

## 快速检查清单

遇到问题时，按以下顺序检查：

- [ ] 是否使用了 `--export=sqlite` 参数？
- [ ] 输出目录是否存在且有写权限？
- [ ] 磁盘空间是否充足？
- [ ] 命令是否成功执行（退出码为 0）？
- [ ] 文件是否在预期位置？
- [ ] SQLite 文件格式是否正确？

---

## 获取帮助

如果以上方法都无法解决问题：

1. **查看 nsys 帮助：**
   ```bash
   nsys profile --help
   nsys export --help
   ```

2. **查看版本信息：**
   ```bash
   nsys --version
   ```

3. **检查日志：**
   ```bash
   nsys profile --output=profile.nsys-rep python script.py 2>&1 | tee nsys.log
   ```

4. **验证环境：**
   ```bash
   which nsys
   echo $CUDA_HOME
   nvidia-smi
   ```

