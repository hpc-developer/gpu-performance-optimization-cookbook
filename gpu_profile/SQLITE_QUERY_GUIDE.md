# SQLite 文件查询指南

## 重要说明

SQLite 文件中的表名和结构可能与预期不同。本指南说明如何正确查询 NVTX 数据。

## 关键表名

### 正确的表名
- ✅ `StringIds` - 存储所有字符串（包括 NVTX 标记名称）
- ✅ `NVTX_EVENTS` - 存储 NVTX 事件数据

### 错误的表名（不存在）
- ❌ `STRING_TABLE` - 不存在，应该使用 `StringIds`
- ❌ `NVTX_TABLE` - 不存在，应该使用 `NVTX_EVENTS`

## 快速查询

### 1. 检查 NVTX 数据是否存在

```bash
# 检查 NVTX_EVENTS 表是否存在
sqlite3 profile.sqlite "SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS';"

# 统计 NVTX 事件数量
sqlite3 profile.sqlite "SELECT COUNT(*) FROM NVTX_EVENTS;"
```

### 2. 查看所有 NVTX 标记名称

```bash
# 从 StringIds 表中查找标记名称
sqlite3 profile.sqlite "SELECT id, value FROM StringIds WHERE value LIKE '%数据%' OR value LIKE '%计算%';"

# 查看所有标记名称（前 20 条）
sqlite3 profile.sqlite "SELECT id, value FROM StringIds LIMIT 20;"
```

### 3. 查看 NVTX 事件数据

```bash
# 查看 NVTX 事件（前 10 条）
sqlite3 profile.sqlite "SELECT * FROM NVTX_EVENTS LIMIT 10;"

# 查看带标记名称的事件（关联 StringIds）
sqlite3 profile.sqlite "
SELECT 
    e.start,
    e.end,
    e.eventType,
    s.value as markName,
    e.text
FROM NVTX_EVENTS e
LEFT JOIN StringIds s ON e.textId = s.id
LIMIT 10;
"
```

### 4. 查找特定标记的事件

```bash
# 查找包含"数据"的标记
sqlite3 profile.sqlite "
SELECT 
    e.start,
    e.end,
    s.value as markName
FROM NVTX_EVENTS e
JOIN StringIds s ON e.textId = s.id
WHERE s.value LIKE '%数据%'
LIMIT 10;
"
```

## 表结构说明

### StringIds 表
```sql
CREATE TABLE StringIds (
    id      INTEGER PRIMARY KEY,  -- 字符串 ID
    value   TEXT                  -- 字符串值（标记名称）
);
```

### NVTX_EVENTS 表
```sql
CREATE TABLE NVTX_EVENTS (
    start       INTEGER,    -- 事件开始时间戳 (ns)
    end         INTEGER,    -- 事件结束时间戳 (ns)
    eventType   INTEGER,    -- 事件类型
    textId      INTEGER,    -- 引用 StringIds(id) - 标记名称 ID
    text        TEXT,       -- 显式文本（非注册字符串）
    color       INTEGER,    -- ARGB 颜色值
    ...
);
```

## 常见查询示例

### 查询所有 NVTX 标记及其事件数量

```sql
SELECT 
    s.value as markName,
    COUNT(*) as eventCount
FROM NVTX_EVENTS e
JOIN StringIds s ON e.textId = s.id
GROUP BY s.value
ORDER BY eventCount DESC;
```

### 查询特定时间范围内的事件

```sql
SELECT 
    s.value as markName,
    e.start,
    e.end,
    (e.end - e.start) / 1000000.0 as duration_ms
FROM NVTX_EVENTS e
JOIN StringIds s ON e.textId = s.id
WHERE e.start > 1000000000 AND e.end < 2000000000
ORDER BY e.start;
```

### 查询最耗时的事件

```sql
SELECT 
    s.value as markName,
    (e.end - e.start) / 1000000.0 as duration_ms
FROM NVTX_EVENTS e
JOIN StringIds s ON e.textId = s.id
WHERE e.end IS NOT NULL
ORDER BY duration_ms DESC
LIMIT 10;
```

## 使用提供的脚本

### 检查 SQLite 文件内容

```bash
./check_sqlite_content.sh profile.sqlite
```

### 查询 NVTX 数据

```bash
./query_nvtx_data.sh profile.sqlite
```

## 常见问题

### Q: 为什么找不到 STRING_TABLE？

**A:** 表名是 `StringIds`，不是 `STRING_TABLE`。使用：
```bash
sqlite3 profile.sqlite "SELECT * FROM StringIds LIMIT 10;"
```

### Q: NVTX_EVENTS 表为空？

**A:** 可能的原因：
1. 没有使用 `--trace=nvtx` 参数
2. 代码中没有添加 NVTX 标记
3. 标记没有被正确记录

**解决方法：**
```bash
# 确保使用 --trace=nvtx
nsys profile --trace=cuda,nvtx --export=sqlite --output=profile.nsys-rep python script.py
```

### Q: 如何查看标记的完整信息？

**A:** 使用关联查询：
```sql
SELECT 
    s.value as markName,
    e.start,
    e.end,
    e.color,
    e.eventType
FROM NVTX_EVENTS e
LEFT JOIN StringIds s ON e.textId = s.id
WHERE s.value IS NOT NULL;
```

### Q: SQLite 文件中有数据，但 GUI 中看不到？

**A:** 这是正常的。SQLite 文件包含原始数据，GUI 工具（nsys-ui）提供更好的可视化。建议：
1. 使用 `nsys-ui profile.nsys-rep` 查看时间线
2. 使用 SQLite 进行程序化分析

## 完整示例

```bash
# 1. 生成包含 NVTX 数据的 SQLite 文件
nsys profile \
  --trace=cuda,nvtx \
  --export=sqlite \
  --output=profile.nsys-rep \
  python3 example1_memory_allocation.py

# 2. 检查 NVTX 数据
./query_nvtx_data.sh profile.sqlite

# 3. 查询特定标记
sqlite3 profile.sqlite "
SELECT 
    s.value as markName,
    COUNT(*) as count,
    AVG((e.end - e.start) / 1000000.0) as avg_duration_ms
FROM NVTX_EVENTS e
JOIN StringIds s ON e.textId = s.id
WHERE s.value LIKE '%数据%'
GROUP BY s.value;
"
```

## 提示

- 使用 `nsys-ui profile.nsys-rep` 可以更直观地查看 NVTX 标记
- SQLite 文件适合程序化分析和批量处理
- 如果数据量很大，考虑使用索引优化查询

