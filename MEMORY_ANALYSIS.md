# my_mapdp.py 显存占用分析

基于代码分析，以下是主要的显存占用点（按占用大小排序）：

## 🔴 最大显存占用点

### 1. **距离矩阵计算** (381-389行) - ⭐⭐⭐⭐⭐
```python
# 行381: dist_c_rows [B, N_c, N_node]
dist_c_rows = _global["node_node"].to(torch.float).gather(1, couriers["target"].unsqueeze(-1).expand(-1, -1, n_node))

# 行382: dist_s1 [B, N_c, n_request]
dist_s1 = dist_c_rows.gather(2, requests["from"].unsqueeze(1).expand(-1, n_courier, -1))

# 行386: dist_s3 [B, N_c, n_request]  
dist_s3 = dist_c_rows.gather(2, requests["station2"].unsqueeze(1).expand(-1, n_courier, -1))
```

**显存占用估算** (假设 batch_size=64, n_courier=20, n_node=40, n_request=330):
- `dist_c_rows`: 64 × 20 × 40 × 4 bytes = **204.8 KB** (相对较小)
- `dist_s1`: 64 × 20 × 330 × 4 bytes = **1.65 MB**
- `dist_s3`: 64 × 20 × 330 × 4 bytes = **1.65 MB**
- `dist_s1_masked`, `dist_s3_masked`: 各 **1.65 MB**
- `topk_s1_idx`, `topk_s3_idx`: 各 **1.65 MB** (int64)
- `dist_mask_s1`, `dist_mask_s3`: 各 **1.65 MB** (bool)

**总计约: ~12 MB** (单次前向传播)

**优化建议:**
- 使用 `torch.no_grad()` 包装距离计算（如果不需要梯度）
- 计算完topk后立即释放中间张量
- 考虑使用 `torch.topk(..., out=...)` 避免创建新张量

---

### 2. **Encoder输出和中间激活** (266-291行) - ⭐⭐⭐⭐⭐
```python
# 行266: all_nodes_embd_pre [B, n_courier + 4*n_request + n_drone + 2*n_request, n_embd]
all_nodes_embd_pre = torch.cat(all_nodes_embd_pre_list, dim=1)

# 行291: all_nodes_embd [B, total_nodes, n_embd]
all_nodes_embd = self.encoder.forward(all_nodes_embd_pre, all_nodes_mask, rel_mat)
```

**显存占用估算** (假设 batch_size=64, n_embd=128, total_nodes≈1500):
- `all_nodes_embd_pre`: 64 × 1500 × 128 × 4 bytes = **49.15 MB**
- `all_nodes_embd`: 64 × 1500 × 128 × 4 bytes = **49.15 MB**
- **Encoder内部激活值** (n_enc_block=10层):
  - 每层self-attention: Q, K, V, attn_weights = 4 × 49.15 MB = **196.6 MB**
  - 每层MLP中间层: 64 × 1500 × 512 × 4 bytes = **196.6 MB**
  - 10层总计: ~**3.9 GB** (前向+反向传播时翻倍)

**这是最大的显存占用点！**

**优化建议:**
- 使用梯度检查点 (gradient checkpointing) 减少激活值存储
- 减小 `n_enc_block` (如果可能)
- 减小 `n_embd` (如果可能)
- 使用混合精度训练 (FP16)

---

### 3. **Attention矩阵计算** (365行, 609行) - ⭐⭐⭐⭐
```python
# 行365: u_courier [B, N_c, 4*n_request+1]
u_courier = D * (q_courier @ k_courier_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()

# 行609: u_drone [B, N_d, 2*n_request+1]
u_drone = D * (q_drone @ k_drone_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
```

**显存占用估算**:
- `q_courier`: 64 × 20 × 128 × 4 bytes = **655 KB**
- `k_courier_with_noaction`: 64 × (4×330+1) × 128 × 4 bytes = **4.3 MB**
- `u_courier`: 64 × 20 × 1321 × 4 bytes = **6.76 MB**
- `q_drone`: 64 × 12 × 128 × 4 bytes = **393 KB**
- `k_drone_with_noaction`: 64 × (2×330+1) × 128 × 4 bytes = **2.16 MB**
- `u_drone`: 64 × 12 × 661 × 4 bytes = **2.03 MB**

**总计约: ~16 MB**

**优化建议:**
- 使用 `torch.matmul` 的 `out` 参数避免创建新张量
- 考虑分块计算attention

---

### 4. **Mask张量** (379行, 585行) - ⭐⭐⭐
```python
# 行379: courier_mask [B, N_c, 4*n_request]
courier_mask = torch.ones(batch_size, n_courier, 4 * n_request, device=device, dtype=torch.bool)

# 行585: drone_mask [B, N_d, 2*n_request+1]
drone_mask = torch.ones(batch_size, n_drone, 2 * n_request + 1, device=device, dtype=torch.bool)
```

**显存占用估算**:
- `courier_mask`: 64 × 20 × 1320 × 1 byte = **1.69 MB**
- `drone_mask`: 64 × 12 × 661 × 1 byte = **507 KB**
- 其他mask操作: `stage1_mask`, `stage2_mask`, `stage3_mask` 等

**总计约: ~3-5 MB**

---

### 5. **节点Embedding构建** (253-268行) - ⭐⭐⭐
```python
# 行248-263: 多个dense embedding
dense = type_emb[None, None].expand(batch_size, n_nodes, -1)
# ... 多个中间张量
all_nodes_embd_pre = torch.cat(all_nodes_embd_pre_list, dim=1)
```

**显存占用估算**:
- 每个segment的dense: ~5-10 MB
- `all_nodes_embd_pre_list`: 8个segment × 平均6 MB = **~48 MB**
- `all_nodes_coord`: 64 × 1500 × 2 × 4 bytes = **768 KB**

**总计约: ~50 MB**

---

### 6. **Cross-Attention计算** (354-358行, 549-553行) - ⭐⭐⭐
```python
# 行354-358: courier cross-attention
g_courier = self.cross_attn.forward(
    self.courier_proj_h(h_courier),
    courier_decode_nodes_embd, 
    courier_decode_nodes_embd
)

# 行549-553: drone cross-attention
g_drone = self.cross_attn.forward(
    self.drone_proj_h(h_drone),
    drone_decode_nodes_embd,
    drone_decode_nodes_embd
)
```

**显存占用估算**:
- `h_courier`: 64 × 20 × (20×129+257+128) × 4 bytes ≈ **13.8 MB**
- `courier_decode_nodes_embd`: 64 × 1320 × 128 × 4 bytes = **43.3 MB**
- Cross-attention内部QKV: 3 × 43.3 MB = **130 MB**
- `h_drone`: 64 × 12 × (12×129+257+128) × 4 bytes ≈ **5.2 MB**
- `drone_decode_nodes_embd`: 64 × 660 × 128 × 4 bytes = **21.6 MB**

**总计约: ~200 MB**

---

## 📊 显存占用总结

| 组件 | 单次前向传播 | 反向传播时 | 优化优先级 |
|------|------------|----------|----------|
| Encoder激活值 | ~3.9 GB | ~7.8 GB | 🔴 最高 |
| Cross-Attention | ~200 MB | ~400 MB | 🟠 高 |
| 节点Embedding | ~50 MB | ~100 MB | 🟡 中 |
| Attention矩阵 | ~16 MB | ~32 MB | 🟡 中 |
| 距离矩阵 | ~12 MB | ~24 MB | 🟢 低 |
| Mask张量 | ~5 MB | ~10 MB | 🟢 低 |

**总显存占用估算:**
- 单次前向: ~**4.2 GB**
- 反向传播: ~**8.4 GB**
- 训练时 (batch_size=64): ~**8-10 GB** (包含梯度)

---

## 🎯 优化建议（按效果排序）

### 1. **使用梯度检查点** (Gradient Checkpointing) - 最有效
```python
# 在Encoder中使用
from torch.utils.checkpoint import checkpoint

def forward(self, x, mask=None, rel_mat=None):
    for block in self.blocks:
        x = checkpoint(block, x, mask, rel_mat)  # 减少激活值存储
    return x
```
**预期减少: 50-70% 显存**

### 2. **减小Encoder层数或embedding维度**
- `n_enc_block`: 10 → 6-8
- `n_embd`: 128 → 96
**预期减少: 20-40% 显存**

### 3. **使用混合精度训练** (FP16)
```python
# 在训练脚本中
from torch.cuda.amp import autocast, GradScaler

with autocast():
    values, log_probs, entropy = policy.forward(obs)
```
**预期减少: 40-50% 显存**

### 4. **优化距离计算**
```python
# 使用no_grad和及时释放
with torch.no_grad():
    dist_c_rows = _global["node_node"].to(torch.float).gather(...)
    # 计算后立即使用，不保存
```
**预期减少: ~10 MB**

### 5. **减小batch_size**
- 从64减至32或16
**预期减少: 50% 显存**

---

## 🔍 代码位置索引

- **行381-389**: 距离矩阵计算（Courier）
- **行266-291**: 节点Embedding和Encoder
- **行354-365**: Courier Attention计算
- **行379-437**: Courier Mask构建
- **行549-609**: Drone Attention计算
- **行585-605**: Drone Mask构建






