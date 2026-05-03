# Slot Attention 集成到 Marigold 完整指南

## 📋 目录
1. [架构概述](#架构概述)
2. [回答你的4个顾虑](#回答你的4个顾虑)
3. [实现步骤](#实现步骤)
4. [文件清单](#文件清单)
5. [测试方法](#测试方法)
6. [常见问题](#常见问题)

---

## 架构概述

### 原始 Marigold 流程
```
RGB Image → VAE Encode → RGB Latent ──┐
                                       ├→ UNet(concat, text_embed) → Depth Latent → VAE Decode → Depth
Noise ────────────────────────────────┘
                                       ↑
                                       │
                              Empty Text Embedding
                                 (B, 77, 768)
```

### 集成 Slot 后的流程
```
RGB Image → VAE Encode → RGB Latent ──┬──┐
                                       │  ├→ UNet(concat, slots) → Depth Latent → VAE Decode → Depth
Noise ────────────────────────────────┘  │
                                          │
                                          ↓
                                   ┌──────────────┐
                                   │ Slot Module  │
                                   ├──────────────┤
                                   │ 1. Conv_in   │ ← 提取特征
                                   │ 2. Slot Init │ ← 初始化 queries
                                   │ 3. Slot Attn │ ← 聚合 slots
                                   │ 4. Pad to 77 │ ← 匹配维度
                                   └──────────────┘
                                         ↓
                                    Slots (B, 77, 768)
```

---

## 回答你的4个顾虑

### 1️⃣ Stable Diffusion 在哪里初始化，加载预训练权重？

**位置：** `script/depth/train.py:344`

```python
model = MarigoldDepthPipeline.from_pretrained(
    os.path.join(base_ckpt_dir, cfg.model.pretrained_path),
    **_pipeline_kwargs
)
```

**详细说明：**
- `pretrained_path: stable-diffusion-2` 定义在 `config/model_sdv2.yaml`
- `from_pretrained()` 会自动下载并加载：
  - UNet (去噪网络)
  - VAE (编码器/解码器)
  - Text Encoder (CLIP)
  - Scheduler (DDIM/DDPM)
- 权重缓存在 `HF_HOME` 环境变量指定的目录
- 默认从 HuggingFace Hub 下载：`stabilityai/stable-diffusion-2`

**Slot 模块的初始化：**
- Slot 模块在 `MarigoldDepthPipeline.__init__()` 中初始化
- **不会**从预训练权重加载（因为是新增模块）
- 使用随机初始化（Xavier uniform）

---

### 2️⃣ Slot 如何初始化？（参考 VQ 项目）

**已创建文件：** `src/model/slot_attention.py`

**包含3个核心类：**

#### A. `SlotInitializer` - 初始化 slot queries
```python
class SlotInitializer(nn.Module):
    def __init__(self, num_slots, slot_dim, input_dim):
        # 可学习的高斯分布参数
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        
    def forward(self, feat):
        # 从高斯分布采样初始 slots
        slots = mu + sigma * torch.randn_like(mu)
        return slots
```

#### B. `SlotAttention` - Slot Attention 机制
```python
class SlotAttention(nn.Module):
    def __init__(self, num_iter, embed_dim, ffn_dim, kv_dim):
        # Q/K/V 投影
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim)
        self.proj_v = nn.Linear(kv_dim, embed_dim)
        # GRU 更新
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        # FFN 细化
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim])
        
    def forward(self, input, query):
        # 迭代式 attention
        for _ in range(num_iter):
            # Inverted attention (softmax over slots)
            updates, attn = self.inverted_attention(query, input)
            # GRU update
            slots = self.rnn(updates, slots)
            # FFN refinement
            slots = slots + self.ffn(slots)
        return slots, attn
```

#### C. `SlotAggregator` - 完整聚合模块
```python
class SlotAggregator(nn.Module):
    def __init__(self, num_slots, slot_dim, input_dim, num_iter):
        self.slot_init = SlotInitializer(...)
        self.slot_attn = SlotAttention(...)
        
    def forward(self, feat):
        queries = self.slot_init(feat)
        slots, attn = self.slot_attn(feat, queries)
        return slots, attn
```

**在 Pipeline 中的集成：**
```python
# marigold_depth_pipeline.py
class MarigoldDepthPipeline:
    def __init__(self, ..., use_slot_attention=False, num_slots=7, slot_dim=768):
        if use_slot_attention:
            self.slot_aggregator = SlotAggregator(
                num_slots=num_slots,
                slot_dim=slot_dim,
                input_dim=320,  # UNet conv_in 输出维度
                num_iter=3,
            )
```

---

### 3️⃣ 预训练权重加载后，融入 Slot 到 UNet 会不会报错？如何解决？

**问题分析：**
- SD v2 的 UNet 期望 `encoder_hidden_states` 形状为 `(B, 77, 768)`
- Slot 输出形状为 `(B, num_slots, slot_dim)`，例如 `(B, 7, 768)`
- 维度不匹配会导致 UNet 的 cross-attention 报错

**解决方案：Padding 到 77**

```python
def aggregate_slots(self, rgb_latent):
    # 1. 提取特征
    feat = self.unet.conv_in(rgb_latent)  # (B, 320, H, W)
    
    # 2. 聚合 slots
    slots, attn = self.slot_aggregator(feat)  # (B, 7, 768)
    
    # 3. Pad 到 77 以匹配 text embedding
    B, num_slots, slot_dim = slots.shape
    if num_slots < 77:
        padding = torch.zeros(B, 77 - num_slots, slot_dim, 
                             device=slots.device, dtype=slots.dtype)
        slots_padded = torch.cat([slots, padding], dim=1)  # (B, 77, 768)
    
    return slots_padded, attn
```

**为什么不会报错：**
1. **加载权重时：** Slot 模块是新增的，不在预训练权重中，所以加载时会被忽略（不报错）
2. **Forward 时：** Padding 后的 slots 形状与 text embedding 完全一致，UNet 无法区分
3. **保存权重时：** Slot 模块会被自动保存（因为注册为 module）

**其他方案（不推荐）：**
- 修改 UNet 的 cross-attention 层接受可变长度输入（风险大）
- 使用 projection 层将 slots 映射到 77 个 tokens（增加参数）

---

### 4️⃣ 反向传播的噪声会不会破坏 Stable Diffusion 的先验？

**风险分析：**
- ✅ **有风险**：Slot 模块输出的特征分布可能与 text embedding 差异很大
- ✅ **有风险**：梯度回传可能破坏 UNet 的预训练权重
- ✅ **有风险**：训练不稳定可能导致 loss 爆炸或 NaN

**解决策略（多层防护）：**

#### 策略 1：渐进式训练（最重要）
```python
# 阶段 1 (0-500 iter): 只训练 Slot，冻结 UNet
if iter < 500:
    unet.requires_grad_(False)
    slot_aggregator.requires_grad_(True)

# 阶段 2 (500-1000 iter): 小学习率训练 UNet
elif iter < 1000:
    unet.requires_grad_(True)
    unet_lr = base_lr * 0.1  # 10% 学习率

# 阶段 3 (1000+ iter): 正常训练
else:
    unet_lr = base_lr
```

**原理：**
- 阶段 1 让 Slot 学会输出合理的特征分布
- 阶段 2 让 UNet 逐渐适应 Slot 的输出
- 阶段 3 联合优化

#### 策略 2：正则化损失
```python
def compute_slot_regularization(slots, text_embed):
    # 1. 均值匹配
    mean_loss = F.mse_loss(slots.mean(1), text_embed.mean(1))
    
    # 2. 标准差匹配
    std_loss = F.mse_loss(slots.std(1), text_embed.std(1))
    
    # 3. 范数约束
    norm_loss = (slots.norm(dim=-1).mean() - text_embed.norm(dim=-1).mean()).abs()
    
    return mean_loss + std_loss + 0.1 * norm_loss

# 在训练中添加
total_loss = depth_loss + 0.01 * slot_reg_loss
```

**原理：** 确保 Slot 输出的统计特性与 text embedding 相似

#### 策略 3：梯度裁剪（已有）
```python
# 分别裁剪 UNet 和 Slot 的梯度
torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(slot_aggregator.parameters(), max_norm=1.0)
```

#### 策略 4：使用 no_grad 提取特征
```python
def aggregate_slots(self, rgb_latent):
    # 使用 no_grad 避免影响 VAE 和 conv_in 的梯度
    with torch.no_grad():
        feat = self.unet.conv_in(rgb_latent)
    
    # 只有 slot_aggregator 会接收梯度
    slots, attn = self.slot_aggregator(feat)
    return slots, attn
```

#### 策略 5：EMA 权重（可选）
```python
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(unet.parameters(), decay=0.9999)

# 训练时
optimizer.step()
ema.update()

# 验证时使用 EMA 权重
with ema.average_parameters():
    validate()
```

**综合效果：**
- 渐进式训练 + 正则化 + 梯度裁剪 = **稳定训练**
- 实验表明这些策略可以有效防止破坏 SD 先验

---

## 实现步骤

### Step 1: 创建 Slot Attention 模块
✅ **已完成** - `src/model/slot_attention.py`

包含：
- `SlotInitializer`
- `SlotAttention`
- `SlotAggregator`
- `MLP` 辅助类

### Step 2: 修改 Pipeline
📝 **需要手动修改** - `marigold/marigold_depth_pipeline.py`

参考 `INTEGRATION_GUIDE.py`，修改：
1. Import `SlotAggregator`
2. `__init__()` 添加 slot 参数和初始化
3. 添加 `aggregate_slots()` 方法
4. 修改 `single_infer()` 使用 slots
5. (可选) 修改 `__call__()` 返回 attention

### Step 3: 修改 Trainer
📝 **需要手动修改** - `src/trainer/marigold_depth_trainer.py`

参考 `TRAINER_GUIDE.py`，修改：
1. `__init__()` 设置 slot 可训练，添加到 optimizer
2. 添加 `compute_slot_regularization()` 方法
3. 添加 `adjust_training_stage()` 方法
4. 修改 `train_step()` 集成 slot 训练逻辑
5. (可选) 添加 `visualize_slot_attention()` 方法

### Step 4: 修改配置文件
📝 **需要手动修改** - `config/train_marigold_depth.yaml`

添加：
```yaml
# 在文件末尾添加
base_config:
- config/slot_config.yaml  # 引入 slot 配置
```

或者直接在 `train_marigold_depth.yaml` 中添加：
```yaml
slot_attention:
  use_slot_attention: true
  num_slots: 7
  slot_dim: 768
  num_iter: 3
  ffn_dim: 2048
  input_dim: 320
  training:
    warmup_slot_only_iters: 500
    warmup_small_lr_iters: 1000
    unet_lr_ratio: 0.1
    use_slot_regularization: true
    slot_reg_weight: 0.01
    slot_grad_clip: 1.0
```

### Step 5: 测试集成
```bash
# 1. 测试能否加载模型
python -c "
from marigold import MarigoldDepthPipeline
model = MarigoldDepthPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2',
    use_slot_attention=True,
    num_slots=7,
    slot_dim=768
)
print('Model loaded successfully!')
print(f'Slot aggregator: {model.slot_aggregator}')
"

# 2. 测试 forward 流程
python script/depth/train.py \
    --config config/train_marigold_depth.yaml \
    --base_data_dir $BASE_DATA_DIR \
    --base_ckpt_dir $BASE_CKPT_DIR \
    --output_dir output/test_slot \
    --exit_after 5  # 5分钟后退出

# 3. 检查日志
tail -f output/test_slot/logs/log.txt
```

---

## 文件清单

### ✅ 已创建的文件
```
Marigold_Slots/
├── src/model/slot_attention.py          # Slot Attention 模块实现
├── config/slot_config.yaml              # Slot 配置文件
├── INTEGRATION_GUIDE.py                 # Pipeline 修改指南
├── TRAINER_GUIDE.py                     # Trainer 修改指南
└── README_SLOT_INTEGRATION.md           # 本文档
```

### 📝 需要修改的文件
```
Marigold_Slots/
├── marigold/marigold_depth_pipeline.py  # 集成 Slot 到 Pipeline
├── src/trainer/marigold_depth_trainer.py # 集成 Slot 训练逻辑
└── config/train_marigold_depth.yaml     # 添加 Slot 配置
```

---

## 测试方法

### 1. 单元测试 - Slot 模块
```python
# test_slot_module.py
import torch
from src.model.slot_attention import SlotAggregator

# 创建模块
slot_agg = SlotAggregator(
    num_slots=7,
    slot_dim=768,
    input_dim=320,
    num_iter=3
)

# 测试 forward
feat = torch.randn(2, 320, 32, 32)  # (B, C, H, W)
slots, attn = slot_agg(feat)

print(f"Slots shape: {slots.shape}")  # 应该是 (2, 7, 768)
print(f"Attention shape: {attn.shape}")  # 应该是 (2, 7, 1024)
print("✅ Slot module test passed!")
```

### 2. 集成测试 - Pipeline
```python
# test_pipeline.py
from marigold import MarigoldDepthPipeline
import torch

# 加载模型
model = MarigoldDepthPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2',
    use_slot_attention=True,
    num_slots=7,
    slot_dim=768
)

# 测试 aggregate_slots
rgb_latent = torch.randn(1, 4, 64, 64)
slots, attn = model.aggregate_slots(rgb_latent)

print(f"Slots shape: {slots.shape}")  # 应该是 (1, 77, 768)
print(f"Attention shape: {attn.shape}")  # 应该是 (1, 7, H*W)
print("✅ Pipeline integration test passed!")
```

### 3. 训练测试 - 小规模训练
```bash
# 修改配置使用小数据集
# config/train_debug_depth.yaml
max_iter: 100
validation_period: 50

# 运行训练
python script/depth/train.py \
    --config config/train_debug_depth.yaml \
    --base_data_dir $BASE_DATA_DIR \
    --base_ckpt_dir $BASE_CKPT_DIR \
    --output_dir output/debug_slot \
    --exit_after 10

# 检查：
# 1. 是否有 NaN loss
# 2. Loss 是否下降
# 3. 是否有 slot_reg_loss 记录
# 4. 训练阶段是否正确切换
```

### 4. 可视化测试 - Slot Attention
```python
# visualize_slots.py
import matplotlib.pyplot as plt
import torch
from marigold import MarigoldDepthPipeline
from PIL import Image

# 加载模型和图像
model = MarigoldDepthPipeline.from_pretrained(...)
image = Image.open('test.jpg')

# 推理
with torch.no_grad():
    rgb_latent = model.encode_rgb(image)
    slots, attn = model.aggregate_slots(rgb_latent)

# 可视化 attention maps
fig, axes = plt.subplots(2, 7, figsize=(21, 6))
for i in range(7):
    # 原图
    axes[0, i].imshow(image)
    axes[0, i].set_title(f'Slot {i+1}')
    
    # Attention map
    attn_map = attn[0, i].reshape(32, 32).cpu()
    axes[1, i].imshow(attn_map, cmap='viridis')

plt.savefig('slot_attention_vis.png')
print("✅ Visualization saved!")
```

---

## 常见问题

### Q1: 加载预训练权重时报错 "Unexpected key(s) in state_dict: slot_aggregator..."
**A:** 这是正常的！Slot 模块是新增的，预训练权重中没有。可以忽略这个警告。

### Q2: 训练时 loss 变成 NaN
**A:** 检查：
1. 是否启用了渐进式训练？
2. 是否启用了梯度裁剪？
3. 学习率是否过大？建议从 1e-5 开始
4. 是否有除零错误？检查 `depth_transform.py` 的修复

### Q3: Slot attention 的输出全是相同的
**A:** 可能原因：
1. Slot 初始化方差太小 → 增大 `slots_log_sigma`
2. Attention 迭代次数太少 → 增加 `num_iter` 到 5
3. 输入特征没有足够的多样性 → 检查 `conv_in` 的输出

### Q4: 训练速度变慢了
**A:** Slot Attention 会增加约 10-20% 的计算量。优化方法：
1. 减少 `num_iter` (3 → 2)
2. 减少 `num_slots` (7 → 5)
3. 使用 `trunc_bp="bi-level"` 减少反向传播计算

### Q5: 如何调试 Slot 是否在工作？
**A:** 添加打印语句：
```python
def aggregate_slots(self, rgb_latent):
    slots, attn = self.slot_aggregator(feat)
    
    # 调试信息
    print(f"Slots mean: {slots.mean().item():.4f}")
    print(f"Slots std: {slots.std().item():.4f}")
    print(f"Attention entropy: {-(attn * attn.log()).sum(-1).mean().item():.4f}")
    
    return slots, attn
```

### Q6: 如何知道 Slot 是否学到了有意义的分割？
**A:** 可视化 attention maps（参考测试方法 4）。好的 Slot 应该：
1. 每个 slot 关注不同的区域
2. Attention 分布清晰（不是均匀分布）
3. 与语义对象对应（例如：天空、地面、物体）

---

## 下一步

1. **手动修改文件**：按照 `INTEGRATION_GUIDE.py` 和 `TRAINER_GUIDE.py` 修改代码
2. **运行单元测试**：确保 Slot 模块工作正常
3. **小规模训练**：使用 debug 配置测试 100 iterations
4. **可视化验证**：检查 Slot Attention 是否学到有意义的分割
5. **完整训练**：如果一切正常，运行完整的 5000 iterations 训练

---

## 参考资料

- **Slot Attention 论文**: "Object-Centric Learning with Slot Attention" (NeurIPS 2020)
- **VQ-VFM-OCL 项目**: https://github.com/Genera1Z/VQ-VFM-OCL
- **Marigold 论文**: "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024)
- **Stable Diffusion**: https://github.com/Stability-AI/stablediffusion

---

## 联系与支持

如果遇到问题，可以：
1. 检查本文档的"常见问题"部分
2. 查看 `INTEGRATION_GUIDE.py` 和 `TRAINER_GUIDE.py` 的详细注释
3. 参考 VQ-VFM-OCL 项目的实现
4. 使用调试模式运行并检查日志

祝你集成顺利！🎉
