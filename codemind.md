# Marigold_Slots 项目详细代码解读

> 本文档全面解析 Marigold_Slots 项目的架构设计、代码实现、数据流和训练策略。
> Marigold_Slots 是在 Marigold 深度估计模型基础上集成 Slot Attention 机制的创新项目。

---

## 目录

1. [项目概述与动机](#1-项目概述与动机)
2. [项目结构](#2-项目结构)
3. [核心架构设计](#3-核心架构设计)
4. [Slot Attention 模块详解](#4-slot-attention-模块详解)
5. [Pipeline 管线详解](#5-pipeline-管线详解)
6. [训练器 (Trainer) 详解](#6-训练器-trainer-详解)
7. [数据管线](#7-数据管线)
8. [损失函数与评估指标](#8-损失函数与评估指标)
9. [配置系统](#9-配置系统)
10. [推理流程](#10-推理流程)
11. [与原版 Marigold 的关键差异](#11-与原版-marigold-的关键差异)
12. [与 VQ-VFM-OCL 的设计对比](#12-与-vq-vfm-ocl-的设计对比)
13. [训练策略与技巧](#13-训练策略与技巧)
14. [完整数据流图](#14-完整数据流图)

---

## 1. 项目概述与动机

### 1.1 背景

**Marigold** 是基于 Stable Diffusion 的单目深度估计模型。它利用预训练扩散模型的丰富视觉先验知识，通过微调实现高质量的深度预测。原版 Marigold 使用空文本嵌入（empty text embedding）作为 UNet 的交叉注意力条件，没有利用图像的语义信息。

**Slot Attention** 是一种以对象为中心（object-centric）的表示学习机制。它能将输入特征自动聚合为固定数量的"槽"（slots），每个槽可以表示场景中的一个对象或语义区域。

### 1.2 动机

将 Slot Attention 集成到 Marigold 中的动机：

1. **增强语义理解**：原版 Marigold 的交叉注意力条件是"空的"，没有任何语义信息。通过 Slot Attention 提取的对象级别特征可以为深度估计提供丰富的语义引导
2. **对象感知的深度估计**：不同对象可能处于不同深度平面。Slot Attention 自然地将场景分解为对象单元，有助于模型理解场景的深度结构
3. **场景分解能力**：Slot Attention 的注意力图可以提供隐式的场景分割，这对于深度估计是有用的辅助信息

### 1.3 核心思路

```
原版 Marigold:  UNet(concat_latent, empty_text_embed) → depth
Marigold_Slots: UNet(concat_latent, slot_features)     → depth
```

用从图像特征中提取的 Slot 特征替代空文本嵌入，作为 UNet 交叉注意力的条件输入。

---

## 2. 项目结构

```
Marigold_Slots/
├── marigold/                              # 推理管线 (HuggingFace diffusers 兼容)
│   ├── marigold_depth_pipeline.py         # 深度估计管线 (547行) ★核心文件
│   ├── marigold_normals_pipeline.py       # 法线估计管线
│   ├── marigold_iid_pipeline.py           # 内在图像分解管线
│   └── util/                              # 工具函数
│       ├── image_util.py                  # 图像处理 (resize, normalize, colorize)
│       ├── batchsize.py                   # 自动batch大小估计
│       └── ensemble.py                    # 多次推理集成
│
├── src/                                   # 训练基础设施
│   ├── model/
│   │   └── slot_attention.py              # Slot Attention 模块 (300行) ★核心文件
│   ├── trainer/
│   │   ├── marigold_depth_trainer.py      # 深度训练器 (717行) ★核心文件
│   │   ├── marigold_iid_trainer.py        # IID训练器
│   │   └── marigold_normals_trainer.py    # 法线训练器
│   ├── dataset/                           # 数据集加载器
│   │   ├── base_depth_dataset.py          # 数据集基类
│   │   ├── hypersim_dataset.py            # Hypersim 合成数据
│   │   ├── kitti_dataset.py               # KITTI 真实数据
│   │   ├── vkitti_dataset.py              # Virtual KITTI
│   │   ├── nyu_dataset.py                 # NYU Depth v2
│   │   └── mixed_sampler.py              # 多数据集混合采样
│   └── util/
│       ├── loss.py                        # 损失函数
│       ├── metric.py                      # 评估指标
│       ├── depth_transform.py             # 深度归一化
│       └── lr_scheduler.py                # 学习率调度
│
├── config/                                # YAML 配置文件
│   ├── train_marigold_depth.yaml          # 主训练配置
│   ├── slot_config.yaml                   # Slot 专用配置
│   ├── model_sdv2.yaml                    # Stable Diffusion v2 模型配置
│   └── dataset_depth/                     # 数据集配置目录
│
├── script/                                # 入口脚本
│   ├── depth/
│   │   ├── train.py                       # 训练入口
│   │   ├── infer.py                       # 推理入口
│   │   ├── eval.py                        # 评估入口
│   │   └── run.py                         # 用户友好CLI
│   ├── normals/
│   └── iid/
│
└── 文档文件
    ├── INTEGRATION_GUIDE.py               # 管线修改指南
    ├── TRAINER_GUIDE.py                   # 训练器修改指南
    ├── README_SLOT_INTEGRATION.md         # 集成文档
    └── QUICK_REFERENCE.txt                # 快速参考卡
```

---

## 3. 核心架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Marigold_Slots 架构                          │
│                                                                  │
│  RGB Image (B,3,H,W)                                            │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐     rgb_latent                                     │
│  │   VAE   │────────────────► (B, 4, h, w)                      │
│  │ Encoder │                      │                              │
│  │ (frozen)│                      │                              │
│  └─────────┘                      │                              │
│                                   │                              │
│  Depth GT (B,1,H,W)              │                              │
│       │                           │                              │
│       ▼                           │                              │
│  ┌─────────┐     gt_latent       │                              │
│  │   VAE   │──────────┐          │                              │
│  │ Encoder │          │          │                              │
│  │ (frozen)│          ▼          │                              │
│  └─────────┘   ┌──────────┐     │                              │
│                │ Add Noise │     │                              │
│                │  t~U(0,T) │     │                              │
│                └─────┬─────┘     │                              │
│                      │           │                              │
│                      ▼           ▼                              │
│                ┌──────────────────────┐                          │
│                │     Concatenate      │                          │
│                │   (B, 8, h, w)       │                          │
│                └──────────┬───────────┘                          │
│                           │                                      │
│           ┌───────────────┼───────────────┐                     │
│           │               │               │                     │
│           ▼               ▼               │                     │
│    ┌────────────┐  ┌──────────────┐       │                     │
│    │ UNet       │  │    Slot      │       │                     │
│    │ conv_in    │  │  Aggregator  │       │                     │
│    │ (8→320)    │  │              │       │                     │
│    └─────┬──────┘  │ ┌──────────┐│       │                     │
│          │         │ │ Init     ││       │                     │
│          │         │ │ Slots    ││       │                     │
│          │         │ └────┬─────┘│       │                     │
│          │         │      ▼      │       │                     │
│          └────────►│ ┌──────────┐│       │                     │
│           feat     │ │  Slot    ││       │                     │
│        (B,320,h,w) │ │ Attention││       │                     │
│                    │ │ (3 iter) ││       │                     │
│                    │ └────┬─────┘│       │                     │
│                    │      │      │       │                     │
│                    │      ▼      │       │                     │
│                    │ ┌──────────┐│       │                     │
│                    │ │ Pad to   ││       │                     │
│                    │ │ (B,77,D) ││       │                     │
│                    │ └────┬─────┘│       │                     │
│                    └──────┼──────┘       │                     │
│                           │              │                     │
│                           ▼              │                     │
│                    slots_padded          │                     │
│                    (B, 77, 1024)         │                     │
│                           │              │                     │
│                           ▼              ▼                     │
│                    ┌──────────────────────────┐                 │
│                    │         UNet             │                 │
│                    │  spatial: cat_latents    │                 │
│                    │  cross_attn: slots       │                 │
│                    └──────────┬───────────────┘                 │
│                               │                                 │
│                               ▼                                 │
│                        noise_pred (B, 4, h, w)                  │
│                               │                                 │
│                               ▼                                 │
│                    ┌──────────────────┐                          │
│                    │    MSE Loss      │                          │
│                    │ pred vs target   │                          │
│                    └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

#### 3.2.1 特征提取来源

Slot Attention 的输入特征来自 **UNet 的 conv_in 层**：

```python
# src/trainer/marigold_depth_trainer.py
feat = self.model.unet.conv_in(cat_latents)  # (B, 320, H, W)
slots, attn = self.model.aggregator(feat)     # (B, num_slots, slot_dim)
```

选择 `conv_in` 输出作为特征源的原因：
- `conv_in` 是 UNet 的第一个卷积层，将 8 通道输入映射到 320 通道
- 这些特征同时包含 RGB 和深度噪声信息
- 320 维特征空间足够丰富，能够支持有效的 slot 聚合

#### 3.2.2 Slot 作为交叉注意力条件

Slots 被填充到 `(B, 77, D)` 的形状，直接替代原版 Marigold 的空文本嵌入：

```python
# Padding: (B, num_slots, D) → (B, 77, D)
padding = torch.zeros(B, 77 - num_slots, slot_dim)
slots_padded = torch.cat([slots, padding], dim=1)  # (B, 77, 1024)

# 替代原来的 empty_text_embed
model_pred = unet(cat_latents, timesteps, slots_padded)
```

为什么是 77 和 1024：
- **77**：Stable Diffusion 的文本编码器输出序列长度为 77 tokens
- **1024**：Stable Diffusion v2 使用的文本编码器（OpenCLIP ViT-H）输出维度为 1024

#### 3.2.3 conv_in 层的修改

原始 Stable Diffusion 的 `conv_in` 接受 4 通道输入。Marigold 需要 8 通道（RGB latent + depth latent），因此需要修改：

```python
# src/trainer/marigold_depth_trainer.py, _replace_unet_conv_in()
def _replace_unet_conv_in(self):
    _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = self.model.unet.conv_in.bias.clone()      # [320]
    _weight = _weight.repeat((1, 2, 1, 1))            # [320, 8, 3, 3]
    _weight *= 0.5  # 保持激活值量级不变
    _new_conv_in = Conv2d(8, 320, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    self.model.unet.conv_in = _new_conv_in
```

**关键细节**：
- 权重复制：将原始 4 通道权重复制一份，扩展到 8 通道
- 乘以 0.5：因为权重翻倍，为保持输出激活值量级不变，需要减半
- 前 4 通道处理 RGB latent，后 4 通道处理 noisy depth latent

---

## 4. Slot Attention 模块详解

> 文件：`src/model/slot_attention.py`

### 4.1 SlotInitializer - 槽初始化器

```python
class SlotInitializer(nn.Module):
    """
    从高斯分布初始化 slot 查询向量
    可选择性地根据输入特征进行条件化
    """
    def __init__(self, num_slots, slot_dim, condition_on_input=True, input_dim=None):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # 可学习的高斯参数
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # 可选：根据输入特征条件化
        if condition_on_input:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim or slot_dim, slot_dim),
                nn.ReLU(),
                nn.Linear(slot_dim, slot_dim),
            )

    def forward(self, features=None, batch_size=None):
        """
        输入: features (B, N, C) 或 batch_size (int)
        输出: slots (B, num_slots, slot_dim)
        """
        B = features.shape[0] if features is not None else batch_size

        # 从学习到的高斯分布采样
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(sigma)
        # slots: (B, num_slots, slot_dim)

        # 如果有输入特征，用其条件化 slots
        if features is not None and hasattr(self, 'input_proj'):
            # 全局平均池化获取图像级特征
            feat_mean = features.mean(dim=1)  # (B, C)
            feat_proj = self.input_proj(feat_mean)  # (B, slot_dim)
            slots = slots + feat_proj.unsqueeze(1)  # 广播加到每个 slot

        return slots
```

**输入/输出**：
| 参数 | 形状 | 说明 |
|------|------|------|
| features (输入) | `(B, N, C)` | 可选的条件特征 |
| slots (输出) | `(B, num_slots, slot_dim)` | 初始化的 slot 查询向量 |

### 4.2 SlotAttention - 槽注意力核心

```python
class SlotAttention(nn.Module):
    """
    迭代式 Slot Attention 机制
    使用"反转注意力"(inverted attention)将特征聚合到 slots
    """
    def __init__(self, num_iter, embed_dim, ffn_dim=None, eps=1e-8):
        super().__init__()
        self.num_iter = num_iter
        self.embed_dim = embed_dim
        self.eps = eps
        ffn_dim = ffn_dim or 4 * embed_dim

        # 层归一化
        self.norm_input = nn.LayerNorm(embed_dim)
        self.norm_slots = nn.LayerNorm(embed_dim)
        self.norm_pre_ff = nn.LayerNorm(embed_dim)

        # Q/K/V 投影
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # GRU 更新门
        self.gru = nn.GRUCell(embed_dim, embed_dim)

        # FFN 精炼
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, inputs, slots):
        """
        输入:
            inputs: (B, N, C) - 编码特征 (来自 conv_in 输出)
            slots:  (B, S, C) - 初始 slot 查询

        输出:
            slots:  (B, S, C) - 更新后的 slots
            attn:   (B, S, N) - 注意力权重图
        """
        B, N, C = inputs.shape
        S = slots.shape[1]

        # 输入特征归一化 & KV 投影 (只做一次)
        inputs_norm = self.norm_input(inputs)
        k = self.proj_k(inputs_norm)  # (B, N, C)
        v = self.proj_v(inputs_norm)  # (B, N, C)

        # 迭代式 slot 更新
        for _ in range(self.num_iter):
            slots_prev = slots

            # Q 投影
            slots_norm = self.norm_slots(slots)
            q = self.proj_q(slots_norm)  # (B, S, C)

            # ★ 反转注意力 (Inverted Attention) ★
            # 标准注意力: softmax 在 key 维度 → 每个 query 关注不同 key
            # 反转注意力: softmax 在 slot 维度 → 每个像素被分配到一个 slot
            scale = C ** -0.5
            attn_logits = torch.einsum("bsc,bnc->bsn", q, k) * scale  # (B, S, N)
            attn = F.softmax(attn_logits, dim=1)  # ★ dim=1 是 slot 维度！

            # 归一化注意力权重 (防止某些 slot 接收过多信息)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

            # 加权聚合
            updates = torch.einsum("bsn,bnc->bsc", attn_norm, v)  # (B, S, C)

            # GRU 更新
            slots = self.gru(
                updates.reshape(-1, C),      # (B*S, C)
                slots_prev.reshape(-1, C),   # (B*S, C)
            ).reshape(B, S, C)              # (B, S, C)

            # FFN 精炼 + 残差连接
            slots = slots + self.ffn(self.norm_pre_ff(slots))

        return slots, attn
```

**反转注意力的关键**：

```
标准注意力 (Transformer):
  softmax 在 key (N) 维度 → attn: (B, S, N)
  含义: 每个 slot 关注哪些像素 (允许重叠)

反转注意力 (Slot Attention):
  softmax 在 slot (S) 维度 → attn: (B, S, N)
  含义: 每个像素被分配到哪个 slot (竞争分配)
```

这确保了像素被近似"排他性地"分配到不同 slots，实现场景分解。

### 4.3 SlotAggregator - 槽聚合器

```python
class SlotAggregator(nn.Module):
    """
    完整的 Slot 聚合管线：初始化 + Slot Attention
    处理输入格式转换，兼容 diffusers 框架
    """
    def __init__(self, num_slots, slot_dim, num_iter, ffn_dim, input_dim):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # 输入投影: conv_in 输出维度 → slot 维度
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, slot_dim),    # 320 → 1024
            nn.LayerNorm(slot_dim),
        )

        # Slot 初始化器
        self.initializer = SlotInitializer(
            num_slots=num_slots,
            slot_dim=slot_dim,
        )

        # Slot Attention 核心
        self.slot_attention = SlotAttention(
            num_iter=num_iter,
            embed_dim=slot_dim,
            ffn_dim=ffn_dim,
        )

    @property
    def dtype(self):
        """diffusers 兼容性"""
        return next(self.parameters()).dtype

    @property
    def device(self):
        """diffusers 兼容性"""
        return next(self.parameters()).device

    def forward(self, features):
        """
        输入: features (B, C, H, W) - 来自 UNet conv_in 的特征
        输出:
            slots: (B, num_slots, slot_dim) - 聚合后的 slot 表示
            attn:  (B, num_slots, H*W) - 注意力权重图
        """
        B, C, H, W = features.shape

        # 空间特征展平 + 维度转换
        feat_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C=320)

        # 投影到 slot 维度
        feat_proj = self.input_proj(feat_flat)  # (B, H*W, slot_dim=1024)

        # 初始化 slots
        slots_init = self.initializer(
            features=feat_proj,
            batch_size=B
        )  # (B, num_slots, slot_dim)

        # 迭代 Slot Attention
        slots, attn = self.slot_attention(
            inputs=feat_proj,
            slots=slots_init,
        )  # slots: (B, num_slots, slot_dim), attn: (B, num_slots, H*W)

        return slots, attn
```

**维度变换流程**：

```
UNet conv_in 输出
    │ (B, 320, H, W)
    ▼
flatten + permute
    │ (B, H*W, 320)
    ▼
Linear(320 → 1024) + LayerNorm
    │ (B, H*W, 1024)
    ▼
SlotInitializer
    │ slots: (B, 16, 1024)
    ▼
SlotAttention × 3 iterations
    │ slots: (B, 16, 1024)
    │ attn:  (B, 16, H*W)
    ▼
输出
```

---

## 5. Pipeline 管线详解

> 文件：`marigold/marigold_depth_pipeline.py`

### 5.1 初始化

```python
class MarigoldDepthPipeline(MarigoldPipeline):
    """
    继承自 MarigoldPipeline，增加 Slot Attention 支持
    兼容 HuggingFace diffusers 的 DiffusionPipeline 接口
    """

    # 声明可选组件，避免 diffusers 验证报错
    _optional_components = ["aggregator"]

    def __init__(
        self,
        unet,
        vae,
        scheduler,
        text_encoder,
        tokenizer,
        # Slot Attention 参数
        num_slots=16,
        slot_dim=1024,
        slot_iter=3,
        slot_ffn_dim=2048,
        slot_input_dim=320,
        aggregator=None,      # 可传入预构建的聚合器
        default_processing_resolution=768,
    ):
        super().__init__(unet, vae, scheduler, text_encoder, tokenizer, ...)

        # 创建或使用传入的 Slot Aggregator
        if aggregator is None:
            self.aggregator = SlotAggregator(
                num_slots=num_slots,
                slot_dim=slot_dim,
                num_iter=slot_iter,
                ffn_dim=slot_ffn_dim,
                input_dim=slot_input_dim,
            )
        else:
            self.aggregator = aggregator

        # 注册为 diffusers 组件
        self.register_modules(aggregator=self.aggregator)
```

### 5.2 RGB 编码

```python
def encode_rgb(self, rgb):
    """
    将 RGB 图像编码到 VAE 潜空间

    输入: rgb (B, 3, H, W) ∈ [-1, 1]
    输出: latent (B, 4, h, w)  其中 h=H/8, w=W/8
    """
    # VAE 编码
    h = self.vae.encoder(rgb)
    moments = self.vae.quant_conv(h)
    mean, logvar = moments.chunk(2, dim=1)

    # 重参数化采样 (训练时)，或直接使用均值 (推理时)
    latent = mean  # 推理时使用均值，训练时会用 sample()

    # 缩放因子 (Stable Diffusion 的标准做法)
    latent = latent * self.vae.config.scaling_factor  # 0.18215

    return latent
```

### 5.3 深度解码

```python
def decode_depth(self, depth_latent):
    """
    将深度潜表示解码为深度图

    输入: depth_latent (B, 4, h, w)
    输出: depth (B, 1, H, W) ∈ [0, 1]
    """
    # 逆缩放
    depth_latent = depth_latent / self.vae.config.scaling_factor

    # VAE 解码
    stacked = self.vae.decode(depth_latent, return_dict=False)[0]
    # stacked: (B, 3, H, W) - VAE 输出 3 通道

    # 取通道平均作为单通道深度
    depth_mean = stacked.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # 裁剪到 [0, 1]
    depth_mean = depth_mean.clip(0, 1)

    return depth_mean
```

### 5.4 单步推理

```python
def single_infer(self, rgb_in, num_inference_steps, show_pbar):
    """
    单次深度推理（可能被多次调用以进行集成）

    输入: rgb_in (B, 3, H, W) ∈ [-1, 1]
    输出: depth_pred (B, 1, H, W)
    """
    device = self.device

    # 1. 编码 RGB
    rgb_latent = self.encode_rgb(rgb_in)  # (B, 4, h, w)

    # 2. 初始化随机噪声作为深度 latent 的起点
    depth_latent = torch.randn_like(rgb_latent)  # (B, 4, h, w)

    # 3. 设置扩散调度器
    self.scheduler.set_timesteps(num_inference_steps, device=device)

    # 4. 去噪循环
    for t in self.scheduler.timesteps:
        # 拼接 RGB latent 和当前深度 latent
        cat_latent = torch.cat([rgb_latent, depth_latent], dim=1)  # (B, 8, h, w)

        # ★ Slot Attention: 提取特征并聚合 slots ★
        feat = self.unet.conv_in(cat_latent)        # (B, 320, h, w)
        slots, attn = self.aggregator(feat)          # (B, 16, 1024)

        # Padding 到 77 tokens
        pad = torch.zeros(B, 77 - slots.shape[1], slots.shape[2], device=device)
        slots_padded = torch.cat([slots, pad], dim=1)  # (B, 77, 1024)

        # UNet 预测噪声
        noise_pred = self.unet(
            cat_latent,
            t,
            encoder_hidden_states=slots_padded,  # ★ 替代 empty_text_embed
        ).sample  # (B, 4, h, w)

        # 调度器去噪一步
        depth_latent = self.scheduler.step(
            noise_pred, t, depth_latent
        ).prev_sample

    # 5. 解码深度
    depth = self.decode_depth(depth_latent)  # (B, 1, H, W)

    return depth
```

---

## 6. 训练器 (Trainer) 详解

> 文件：`src/trainer/marigold_depth_trainer.py`

### 6.1 初始化流程

```python
class MarigoldDepthTrainer:
    def __init__(self, cfg, model, train_loader, val_loader=None):
        self.cfg = cfg
        self.model = model  # MarigoldDepthPipeline 实例

        # ═══ 冻结组件 ═══
        self.model.vae.requires_grad_(False)           # VAE 完全冻结
        self.model.text_encoder.requires_grad_(False)   # 文本编码器完全冻结

        # ═══ 可训练组件 ═══
        self.model.unet.requires_grad_(True)            # UNet 可训练
        self.model.aggregator.requires_grad_(True)      # Slot Aggregator 可训练

        # ═══ conv_in 修改 ═══
        self._replace_unet_conv_in()  # 4通道 → 8通道

        # ═══ 优化器 ═══
        # 组合 UNet 和 Aggregator 的参数
        trainable_params = list(self.model.unet.parameters()) + \
                          list(self.model.aggregator.parameters())
        self.optimizer = Adam(trainable_params, lr=cfg.lr)  # lr=3e-5

        # ═══ 学习率调度 ═══
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=cfg.max_iter,
        )

        # ═══ 噪声调度器 ═══
        self.training_noise_scheduler = DDPMScheduler.from_pretrained(...)

        # ═══ 空文本嵌入 (备用) ═══
        self.empty_text_embed = self._encode_empty_text()  # (1, 77, 1024)
```

### 6.2 训练步骤详解

```python
def training_step(self, batch, step):
    """
    单步训练的完整流程

    batch: {
        'rgb_norm':        (B, 3, H, W) ∈ [-1, 1],
        'depth_raw_norm':  (B, 1, H, W) ∈ [-1, 1],
        'valid_mask_raw':  (B, 1, H, W) ∈ {0, 1},
    }
    """

    # ═══════════════════════════════════════
    # Step 1: 编码到潜空间
    # ═══════════════════════════════════════

    rgb = batch['rgb_norm'].to(self.device)            # (B, 3, H, W)
    depth_gt = batch['depth_raw_norm'].to(self.device)  # (B, 1, H, W)
    valid_mask = batch['valid_mask_raw'].to(self.device) # (B, 1, H, W)

    with torch.no_grad():
        # RGB → VAE 潜空间
        rgb_latent = self.model.encode_rgb(rgb)       # (B, 4, h, w)

        # 深度 GT → VAE 潜空间
        # 深度是单通道，需要复制到 3 通道以匹配 VAE 输入
        depth_gt_3ch = depth_gt.repeat(1, 3, 1, 1)    # (B, 3, H, W)
        gt_latent = self.model.encode_rgb(depth_gt_3ch) # (B, 4, h, w)

    # ═══════════════════════════════════════
    # Step 2: 扩散过程 - 加噪
    # ═══════════════════════════════════════

    # 采样随机时间步
    timesteps = torch.randint(
        0, self.training_noise_scheduler.config.num_train_timesteps,
        (B,), device=self.device
    ).long()

    # 采样噪声 (支持多分辨率噪声)
    noise = self._generate_noise(gt_latent)  # (B, 4, h, w)

    # 向 GT latent 加噪
    noisy_latent = self.training_noise_scheduler.add_noise(
        gt_latent, noise, timesteps
    )  # (B, 4, h, w)

    # ═══════════════════════════════════════
    # Step 3: 拼接输入
    # ═══════════════════════════════════════

    cat_latents = torch.cat([rgb_latent, noisy_latent], dim=1)  # (B, 8, h, w)
    # 前4通道: RGB latent (干净的图像编码)
    # 后4通道: Noisy depth latent (加噪的深度编码)

    # ═══════════════════════════════════════
    # Step 4: Slot Attention 特征聚合  ★
    # ═══════════════════════════════════════

    # 从 UNet conv_in 提取特征
    feat = self.model.unet.conv_in(cat_latents)  # (B, 320, h, w)

    # Slot 聚合
    slots, attn_maps = self.model.aggregator(feat)  # slots: (B, 16, 1024)

    # Pad 到 77 tokens (匹配文本嵌入长度)
    pad_size = 77 - slots.shape[1]  # 77 - 16 = 61
    padding = torch.zeros(
        B, pad_size, slots.shape[2],
        device=self.device, dtype=slots.dtype
    )
    slots_padded = torch.cat([slots, padding], dim=1)  # (B, 77, 1024)

    # ═══════════════════════════════════════
    # Step 5: UNet 前向传播
    # ═══════════════════════════════════════

    model_pred = self.model.unet(
        sample=cat_latents,               # 空间输入 (B, 8, h, w)
        timestep=timesteps,                # 时间步
        encoder_hidden_states=slots_padded, # ★ 交叉注意力条件 (B, 77, 1024)
    ).sample  # (B, 4, h, w)

    # ═══════════════════════════════════════
    # Step 6: 计算损失
    # ═══════════════════════════════════════

    # 预测目标: 噪声 (epsilon 参数化)
    target = noise  # (B, 4, h, w)

    # 有效区域掩码 (下采样到 latent 分辨率)
    if valid_mask is not None:
        valid_mask_latent = F.interpolate(
            valid_mask.float(), size=gt_latent.shape[2:],
            mode='nearest'
        )  # (B, 1, h, w)
        valid_mask_latent = valid_mask_latent > 0.5

        # 只在有效区域计算损失
        latent_loss = F.mse_loss(
            model_pred[valid_mask_latent.expand_as(model_pred)],
            target[valid_mask_latent.expand_as(target)],
        )
    else:
        latent_loss = F.mse_loss(model_pred, target)

    # ═══════════════════════════════════════
    # Step 7: 反向传播与优化
    # ═══════════════════════════════════════

    # 梯度累积
    loss = latent_loss / self.gradient_accumulation_steps
    loss.backward()

    if (step + 1) % self.gradient_accumulation_steps == 0:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.unet.parameters(), max_norm=1.0
        )
        torch.nn.utils.clip_grad_norm_(
            self.model.aggregator.parameters(), max_norm=1.0
        )

        # 优化器步进
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    return {"loss": latent_loss.item()}
```

### 6.3 多分辨率噪声

```python
def _generate_noise(self, latent):
    """
    生成多分辨率噪声 (Multi-resolution noise)
    融合不同尺度的噪声，有助于模型学习全局和局部深度结构

    输入: latent (B, 4, h, w) - 参考形状
    输出: noise (B, 4, h, w) - 多分辨率噪声
    """
    noise = torch.randn_like(latent)  # 基础噪声

    # 多分辨率噪声退火
    if self.multi_res_noise_annealing > 0:
        # 在不同分辨率下生成噪声并混合
        for res in [latent.shape[2] // 2, latent.shape[2] // 4]:
            low_res_noise = torch.randn(
                B, 4, res, res, device=self.device
            )
            upsampled = F.interpolate(
                low_res_noise, size=latent.shape[2:],
                mode='bilinear', align_corners=False
            )
            noise = noise + upsampled * self.multi_res_noise_weight

        # 归一化
        noise = noise / noise.std()

    return noise
```

### 6.4 验证流程

```python
def validate(self, step):
    """验证流程 - 使用少量去噪步骤快速评估"""
    self.model.eval()
    metrics = {}

    with torch.no_grad():
        for batch in self.val_loader:
            rgb = batch['rgb_norm'].to(self.device)
            depth_gt = batch['depth_raw'].to(self.device)

            # 推理 (1步去噪 for 快速验证)
            depth_pred = self.model.single_infer(
                rgb,
                num_inference_steps=1,
                show_pbar=False,
            )

            # 对齐预测深度和GT (最小二乘法)
            depth_pred_aligned = self.align_depth(depth_pred, depth_gt)

            # 计算指标
            abs_rel = compute_abs_rel(depth_pred_aligned, depth_gt)
            metrics['abs_rel'] = abs_rel

    self.model.train()
    return metrics
```

---

## 7. 数据管线

### 7.1 基础数据集

> 文件：`src/dataset/base_depth_dataset.py`

```python
class BaseDepthDataset(Dataset):
    """
    深度估计数据集基类
    处理 RGB-Depth 对的加载、归一化和增强
    """

    class DatasetMode(Enum):
        RGB_ONLY = 0   # 只加载 RGB
        EVAL = 1       # RGB + 原始深度 (不做归一化)
        TRAIN = 2      # RGB + 归一化深度

    def __init__(self, mode, filename_ls_path, dataset_dir, ...):
        self.mode = mode
        self.filenames = self._read_filelist(filename_ls_path)
        self.resize_to_hw = resize_to_hw  # (480, 640) for training

    def __getitem__(self, idx):
        """
        返回:
        {
            'rgb_norm':        (3, H, W) ∈ [-1, 1]     # 归一化 RGB
            'depth_raw_norm':  (1, H, W) ∈ [-1, 1]     # 归一化深度
            'valid_mask_raw':  (1, H, W) ∈ {0, 1}      # 有效深度掩码
            'depth_raw_linear':(1, H, W)                # 原始线性深度 (eval)
        }
        """
        # 加载 RGB
        rasters = self._read_rgb(self.rgb_paths[idx])       # (3, H, W) [0, 255]
        rasters['rgb_norm'] = rasters['rgb_int'] / 127.5 - 1  # → [-1, 1]

        # 加载深度
        if self.mode != self.DatasetMode.RGB_ONLY:
            depth = self._read_depth(self.depth_paths[idx])  # (1, H, W)
            valid_mask = depth > 0                            # (1, H, W)

            if self.mode == self.DatasetMode.TRAIN:
                # 深度归一化到 [-1, 1]
                depth_norm = self._normalize_depth(depth)
                rasters['depth_raw_norm'] = depth_norm
            else:
                rasters['depth_raw_linear'] = depth

            rasters['valid_mask_raw'] = valid_mask.float()

        # 数据增强 (训练时)
        if self.augm_args is not None:
            rasters = self._augment(rasters)

        return rasters

    def _normalize_depth(self, depth):
        """
        深度归一化: 线性深度 → [-1, 1]

        方法: scale-and-shift 归一化
        d_norm = 2 * (d - d_min) / (d_max - d_min) - 1
        """
        valid = depth[depth > 0]
        d_min, d_max = valid.min(), valid.max()
        depth_norm = 2 * (depth - d_min) / (d_max - d_min + 1e-8) - 1
        return depth_norm
```

### 7.2 Hypersim 数据集

```python
class HypersimDataset(BaseDepthDataset):
    """
    Apple Hypersim 合成室内场景数据集
    提供精确的深度标注
    """
    def _read_depth(self, path):
        # 读取 HDF5 格式深度
        with h5py.File(path, 'r') as f:
            depth = f['dataset'][:]  # 线性深度 (米)
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        return depth
```

### 7.3 混合采样器

```python
class MixedBatchSampler:
    """
    多数据集混合采样
    训练配置: 90% Hypersim + 10% vKITTI
    """
    def __init__(self, datasets, probabilities, batch_size):
        self.datasets = datasets
        self.probabilities = probabilities  # [0.9, 0.1]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            # 按概率选择数据集
            ds_idx = np.random.choice(
                len(self.datasets), p=self.probabilities
            )
            # 从选中数据集采样
            indices = np.random.choice(
                len(self.datasets[ds_idx]),
                size=self.batch_size,
                replace=True,
            )
            yield [(ds_idx, idx) for idx in indices]
```

---

## 8. 损失函数与评估指标

### 8.1 损失函数

> 文件：`src/util/loss.py`

```python
# ═══ 主要使用的损失 ═══

class MSELoss:
    """标准均方误差损失 - 用于潜空间噪声预测"""
    def __call__(self, pred, target, mask=None):
        if mask is not None:
            return F.mse_loss(pred[mask], target[mask])
        return F.mse_loss(pred, target)


# ═══ 可选的高级损失 ═══

class SILogMSELoss:
    """尺度不变对数均方误差"""
    def __call__(self, pred, target, mask=None):
        log_diff = torch.log(pred[mask]) - torch.log(target[mask])
        silog = (log_diff ** 2).mean() - (log_diff.mean()) ** 2
        return silog


class MeanAbsRelLoss:
    """平均绝对相对误差"""
    def __call__(self, pred, target, mask=None):
        return (torch.abs(pred[mask] - target[mask]) / target[mask]).mean()
```

**训练损失计算流程**：

```
UNet 输出: model_pred (B, 4, h, w)
              │
              ├─── target = noise (epsilon 参数化)
              │        或
              │    target = gt_latent (sample 参数化)
              │
              ▼
    MSE Loss = ||model_pred - target||²
              │
              ├─── 应用 valid_mask (过滤无效深度区域)
              │
              ├─── 除以 gradient_accumulation_steps
              │
              ▼
    loss.backward()
```

### 8.2 评估指标

> 文件：`src/util/metric.py`

```python
# ═══ 主要指标 ═══

def abs_relative_difference(pred, gt):
    """绝对相对误差 (越小越好)"""
    return (torch.abs(pred - gt) / gt).mean()

def delta1_acc(pred, gt, threshold=1.25):
    """阈值准确率 δ1 (越大越好)"""
    ratio = torch.max(pred / gt, gt / pred)
    return (ratio < threshold).float().mean()

# ═══ 所有支持的指标 ═══

METRICS = {
    'abs_rel':   abs_relative_difference,       # 绝对相对误差
    'sq_rel':    squared_relative_difference,    # 平方相对误差
    'rmse':      rmse_linear,                    # 线性 RMSE
    'rmse_log':  rmse_log,                       # 对数 RMSE
    'log10':     log10_error,                    # 对数10 误差
    'delta1':    delta1_acc,                     # δ < 1.25
    'delta2':    delta2_acc,                     # δ < 1.25²
    'delta3':    delta3_acc,                     # δ < 1.25³
    'i_rmse':    inverse_rmse,                   # 逆深度 RMSE
    'silog':     silog_rmse,                     # 尺度不变对数 RMSE
}
```

**深度对齐方法**：

```python
def align_depth_least_squares(pred, gt, valid_mask):
    """
    最小二乘法对齐预测深度到 GT 深度
    
    因为单目深度估计是尺度模糊的，需要对齐：
    aligned = scale * pred + shift
    
    通过最小二乘求解 scale 和 shift
    """
    valid_pred = pred[valid_mask]
    valid_gt = gt[valid_mask]
    
    # 构建线性方程组 [pred, 1] @ [scale, shift]^T = gt
    A = torch.stack([valid_pred, torch.ones_like(valid_pred)], dim=1)
    b = valid_gt
    
    # 最小二乘求解
    result = torch.linalg.lstsq(A, b.unsqueeze(1))
    scale, shift = result.solution.squeeze()
    
    aligned = scale * pred + shift
    return aligned
```

---

## 9. 配置系统

### 9.1 主训练配置

> 文件：`config/train_marigold_depth.yaml`

```yaml
# ═══ 模型配置 ═══
pipeline:
  name: MarigoldDepthPipeline
  kwargs:
    pretrained_model_name_or_path: "stabilityai/stable-diffusion-2"
    num_slots: 16              # Slot 数量 (场景中的对象数)
    slot_iter: 3               # Slot Attention 迭代次数
    slot_dim: 1024             # Slot 嵌入维度 (匹配 SD v2 文本编码器)
    slot_ffn_dim: 2048         # FFN 隐藏维度
    slot_input_dim: 320        # 输入维度 (UNet conv_in 输出通道数)

# ═══ 训练配置 ═══
max_iter: 5000                 # 总训练迭代数
lr: 3.0e-05                    # 学习率
seed: 2024                     # 随机种子
gradient_accumulation_steps: 4  # 梯度累积步数
effective_batch_size: 8        # 有效 batch size = 2 × 4

# ═══ 损失配置 ═══
loss:
  name: mse_loss               # MSE 损失
  kwargs: {}

# ═══ 调度器配置 ═══
lr_scheduler:
  name: cosine
  kwargs:
    num_warmup_steps: 100

# ═══ 噪声配置 ═══
noise:
  multi_res_noise: true
  multi_res_noise_weight: 0.5
  noise_annealing: true

# ═══ 数据配置 ═══
dataloader:
  batch_size: 2
  num_workers: 2
  pin_memory: true

dataset:
  train:
    - name: hypersim
      weight: 0.9
      kwargs:
        resize_to_hw: [480, 640]
    - name: vkitti
      weight: 0.1
      kwargs:
        resize_to_hw: [480, 640]

# ═══ 验证配置 ═══
validation:
  denoising_steps: 1           # 快速验证 (1步去噪)
  ensemble_size: 1             # 不使用集成
  dataset: nyu_v2

# ═══ 保存配置 ═══
trainer:
  save_period: 50              # 每50步保存
  validation_period: 250       # 每250步验证
  visualization_period: 500    # 每500步可视化
```

### 9.2 Slot 专用配置

> 文件：`config/slot_config.yaml`

```yaml
slot_attention:
  use_slot_attention: true
  num_slots: 7                       # Slot 数量
  slot_dim: 768                      # Slot 维度 (SD v1 用 768)
  num_iter: 3                        # Slot Attention 迭代次数
  ffn_dim: 2048                      # FFN 隐层维度
  input_dim: 320                     # 输入特征维度

  training:
    # ═══ 分阶段训练策略 ═══
    warmup_slot_only_iters: 500      # 阶段1: 只训练 slots (0-500)
    warmup_small_lr_iters: 1000      # 阶段2: 小LR训练 UNet (500-1000)
    unet_lr_ratio: 0.1               # UNet 学习率倍率

    # ═══ Slot 正则化 ═══
    use_slot_regularization: true    # 是否使用 slot 分布正则化
    slot_reg_weight: 0.01            # 正则化权重
    slot_grad_clip: 1.0              # Slot 梯度裁剪
```

---

## 10. 推理流程

### 10.1 完整推理管线

```python
# script/depth/run.py - 用户调用入口

from marigold import MarigoldDepthPipeline

# 1. 加载模型
pipeline = MarigoldDepthPipeline.from_pretrained(
    "path/to/checkpoint",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

# 2. 加载图像
image = Image.open("input.jpg")

# 3. 推理
depth_output = pipeline(
    image,
    denoising_steps=4,             # 去噪步数 (默认4)
    ensemble_size=5,               # 集成次数 (默认5)
    processing_res=768,            # 处理分辨率
    match_input_res=True,          # 输出匹配输入分辨率
    color_map="Spectral",         # 深度可视化颜色映射
)

# 4. 获取结果
depth_np = depth_output.depth_np         # numpy深度图 (H, W) ∈ [0, 1]
depth_colored = depth_output.depth_colored  # 彩色深度图 PIL.Image
```

### 10.2 推理数据流

```
输入 RGB 图像 (H, W, 3)
    │
    ▼
预处理: resize + normalize → (B, 3, 768, 768) ∈ [-1, 1]
    │
    ▼
VAE Encode → rgb_latent (B, 4, 96, 96)
    │
    ▼
初始化随机噪声 → depth_latent (B, 4, 96, 96)
    │
    ▼
╔═══════════════════════════════════════════╗
║  去噪循环 (4步 DDIM)                      ║
║                                           ║
║  for t in [999, 749, 499, 249]:          ║
║    │                                      ║
║    ├─ cat = [rgb_latent, depth_latent]   ║
║    │       (B, 8, 96, 96)                ║
║    │                                      ║
║    ├─ feat = conv_in(cat)                ║
║    │       (B, 320, 96, 96)              ║
║    │                                      ║
║    ├─ slots, attn = aggregator(feat)     ║
║    │       (B, 16, 1024)                 ║
║    │                                      ║
║    ├─ slots_pad → (B, 77, 1024)          ║
║    │                                      ║
║    ├─ noise_pred = unet(cat, t, slots)   ║
║    │       (B, 4, 96, 96)                ║
║    │                                      ║
║    └─ depth_latent = scheduler.step(...)  ║
║           (B, 4, 96, 96)                 ║
╚═══════════════════════════════════════════╝
    │
    ▼
VAE Decode → depth_raw (B, 3, 768, 768)
    │
    ▼
Channel Average → depth (B, 1, 768, 768)
    │
    ▼
Clip [0, 1] + Resize to original → (H, W)
    │
    ▼
集成 (5次推理取中位数/均值)
    │
    ▼
输出深度图 (H, W) ∈ [0, 1]
```

### 10.3 集成推理

```python
def ensemble_depths(depth_preds, method="median"):
    """
    多次推理结果集成
    
    depth_preds: list of (H, W) numpy arrays, 每次推理的结果
    
    步骤:
    1. 对每个预测进行 scale-shift 对齐 (到第一个预测的尺度)
    2. 取中位数或均值
    3. 重新归一化到 [0, 1]
    """
    # 以第一个预测为参考
    ref = depth_preds[0]
    
    # 对齐其他预测到参考尺度
    aligned = [ref]
    for pred in depth_preds[1:]:
        scale, shift = least_squares_align(pred, ref)
        aligned.append(scale * pred + shift)
    
    # 集成
    stacked = np.stack(aligned, axis=0)
    if method == "median":
        result = np.median(stacked, axis=0)
    else:
        result = np.mean(stacked, axis=0)
    
    # 归一化到 [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result
```

---

## 11. 与原版 Marigold 的关键差异

### 11.1 对比总结

| 方面 | 原版 Marigold | Marigold_Slots |
|------|---------------|----------------|
| **条件信号** | 空文本嵌入 `(B, 77, 1024)` | Slot 特征 `(B, 77, 1024)` |
| **语义理解** | 无 (空条件) | 对象级语义 (slot 分解) |
| **特征提取** | 无额外提取 | UNet conv_in → Slot Attention |
| **可训练参数** | UNet | UNet + SlotAggregator |
| **训练策略** | 单阶段 | 三阶段渐进式 |
| **场景分割** | 不提供 | 可通过 attn_maps 获取 |
| **推理开销** | 基准 | +Slot Attention 计算 |
| **依赖** | diffusers | diffusers + slot_attention.py |

### 11.2 代码级别的修改点

```python
# ═══ 修改1: Pipeline 初始化 ═══
# 原版
class MarigoldDepthPipeline:
    def __init__(self, unet, vae, scheduler, text_encoder, tokenizer):
        ...

# Slots 版
class MarigoldDepthPipeline:
    def __init__(self, unet, vae, scheduler, text_encoder, tokenizer,
                 num_slots=16, slot_dim=1024, ...):  # ★ 新增参数
        self.aggregator = SlotAggregator(...)          # ★ 新增组件


# ═══ 修改2: 推理时的条件生成 ═══
# 原版
encoder_hidden_states = self.empty_text_embed.expand(B, -1, -1)
noise_pred = self.unet(cat_latent, t, encoder_hidden_states).sample

# Slots 版
feat = self.unet.conv_in(cat_latent)                    # ★ 新增
slots, attn = self.aggregator(feat)                      # ★ 新增
slots_padded = pad_to_77(slots)                          # ★ 新增
noise_pred = self.unet(cat_latent, t, slots_padded).sample  # ★ 修改条件


# ═══ 修改3: Trainer 优化器 ═══
# 原版
optimizer = Adam(self.model.unet.parameters(), lr=lr)

# Slots 版
trainable_params = list(self.model.unet.parameters()) + \
                   list(self.model.aggregator.parameters())  # ★ 新增
optimizer = Adam(trainable_params, lr=lr)


# ═══ 修改4: conv_in 通道扩展 ═══
# 原版 Stable Diffusion: conv_in(4 → 320)
# Marigold/Marigold_Slots: conv_in(8 → 320)
_weight = unet.conv_in.weight.clone()        # [320, 4, 3, 3]
_weight = _weight.repeat((1, 2, 1, 1))       # [320, 8, 3, 3]
_weight *= 0.5                                # 保持激活量级
```

---

## 12. 与 VQ-VFM-OCL 的设计对比

### 12.1 架构差异

```
VQ-VFM-OCL (SlotDiffusion):
    RGB → DINO ViT → (B, 384, h, w) → Slot Attention → slots (B, 7, 256)
                                                            │
    RGB → VQVAE → quant (B, 4, h, w) ─────────────────────┤
                                                            │
                                               UNet(quant, slots) → decode

Marigold_Slots:
    RGB → VAE → rgb_latent (B, 4, h, w) ──┐
    Depth+Noise → VAE → noisy (B, 4, h, w)─┤
                                            ▼
                                    cat (B, 8, h, w)
                                            │
                                    conv_in (B, 320, h, w)
                                            │
                                    Slot Attention → slots (B, 16, 1024)
                                            │
                                    UNet(cat, slots) → noise_pred
```

### 12.2 关键设计差异

| 设计决策 | VQ-VFM-OCL | Marigold_Slots |
|----------|-----------|----------------|
| **Slot 特征来源** | DINO ViT (独立编码器) | UNet conv_in (共享编码器) |
| **特征维度** | 384 (ViT-S) | 320 (conv_in) |
| **Slot 维度** | 256 | 1024 (匹配 SD v2 文本编码器) |
| **Slot 数量** | 7 | 16 |
| **Slot 条件方式** | UNet 交叉注意力 | UNet 交叉注意力 |
| **空间输入** | VQVAE quant (4-dim) | cat_latents (8-dim) |
| **VAE 类型** | 自训练 VQVAE | 预训练 SD VAE |
| **训练信号** | 重构 + 扩散 | 噪声预测 |
| **特征污染风险** | 低 (独立编码器) | 中 (含噪声信息) |

### 12.3 优劣分析

**VQ-VFM-OCL 的优势**：
- Slot 特征来自干净的 RGB，不受噪声污染
- DINO ViT 提供强大的语义特征
- 特征提取和空间输入完全解耦

**Marigold_Slots 的优势**：
- 不需要额外的特征提取网络
- 利用了 UNet 已有的特征提取能力
- 架构更简洁，参数更少

**Marigold_Slots 的潜在问题**：
- `conv_in` 的输入包含 noisy depth latent，可能影响 slot 质量
- 在去噪后期（噪声较小时）可能效果更好
- 可以考虑只使用前 4 通道（RGB latent）的特征来初始化 slots

---

## 13. 训练策略与技巧

### 13.1 三阶段渐进训练

```
Stage 1: Slot-Only Warmup (0 - 500 iterations)
┌─────────────────────────────────────┐
│ UNet: frozen (requires_grad=False)  │
│ Aggregator: trainable              │
│ 目的: 让 slots 学会有意义的场景分解   │
└─────────────────────────────────────┘
              │
              ▼
Stage 2: Small LR UNet (500 - 1000 iterations)
┌─────────────────────────────────────┐
│ UNet: trainable (lr × 0.1)         │
│ Aggregator: trainable              │
│ 目的: UNet 逐步适应 slot 条件        │
└─────────────────────────────────────┘
              │
              ▼
Stage 3: Full Training (1000+ iterations)
┌─────────────────────────────────────┐
│ UNet: trainable (full lr)           │
│ Aggregator: trainable              │
│ 目的: 联合优化达到最佳效果            │
└─────────────────────────────────────┘
```

### 13.2 梯度管理

```python
# 梯度裁剪 - 分别处理 UNet 和 Aggregator
torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(aggregator.parameters(), max_norm=1.0)

# 梯度累积 - 模拟大 batch size
# effective_batch = batch_size × gradient_accumulation_steps × num_gpus
# 例: 2 × 4 × 1 = 8
```

### 13.3 内存优化

```python
# VAE 编码在 no_grad 下进行 (不需要梯度)
with torch.no_grad():
    rgb_latent = self.model.encode_rgb(rgb)
    gt_latent = self.model.encode_rgb(depth_gt_3ch)

# 混合精度训练 (可选)
with torch.autocast("cuda", dtype=torch.float16):
    model_pred = self.model.unet(cat_latents, timesteps, slots_padded).sample
```

---

## 14. 完整数据流图

### 14.1 训练数据流 (逐维度追踪)

```
╔══════════════════════════════════════════════════════════════════════╗
║                        训练数据流                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  输入数据                                                            ║
║  ├── rgb_norm:       (B=2, 3, 480, 640)    ∈ [-1, 1]               ║
║  ├── depth_raw_norm: (B=2, 1, 480, 640)    ∈ [-1, 1]               ║
║  └── valid_mask:     (B=2, 1, 480, 640)    ∈ {0, 1}                ║
║                                                                     ║
║  VAE 编码 (frozen, no_grad)                                         ║
║  ├── rgb_latent:     (2, 4, 60, 80)        = VAE.encode(rgb)       ║
║  └── gt_latent:      (2, 4, 60, 80)        = VAE.encode(depth×3ch) ║
║                                                                     ║
║  加噪过程                                                            ║
║  ├── noise:          (2, 4, 60, 80)        ~ N(0, 1) + multi_res   ║
║  ├── timesteps:      (2,)                  ~ U(0, 1000)            ║
║  └── noisy_latent:   (2, 4, 60, 80)        = add_noise(gt, noise, t)║
║                                                                     ║
║  拼接                                                                ║
║  └── cat_latents:    (2, 8, 60, 80)        = [rgb_lat, noisy_lat]  ║
║                                                                     ║
║  Slot 聚合                                                           ║
║  ├── feat:           (2, 320, 60, 80)      = conv_in(cat_latents)  ║
║  ├── feat_flat:      (2, 4800, 320)        = flatten + permute     ║
║  ├── feat_proj:      (2, 4800, 1024)       = Linear(320→1024)      ║
║  ├── slots_init:     (2, 16, 1024)         = SlotInitializer()     ║
║  ├── slots:          (2, 16, 1024)         = SlotAttention(×3)     ║
║  ├── attn_maps:      (2, 16, 4800)         = attention weights     ║
║  └── slots_padded:   (2, 77, 1024)         = [slots, zeros(61)]    ║
║                                                                     ║
║  UNet 前向                                                           ║
║  ├── sample:         (2, 8, 60, 80)        = cat_latents           ║
║  ├── timestep:       (2,)                  = timesteps             ║
║  ├── encoder_hidden: (2, 77, 1024)         = slots_padded          ║
║  └── model_pred:     (2, 4, 60, 80)        = UNet output           ║
║                                                                     ║
║  损失计算                                                            ║
║  ├── target:         (2, 4, 60, 80)        = noise (ε-param)       ║
║  ├── valid_mask_lat: (2, 1, 60, 80)        = downsample(mask)      ║
║  └── loss:           scalar                = MSE(pred, target)     ║
║                                                                     ║
║  优化                                                                ║
║  ├── loss / grad_accum_steps → backward()                           ║
║  ├── clip_grad_norm_(unet, 1.0)                                     ║
║  ├── clip_grad_norm_(aggregator, 1.0)                               ║
║  ├── optimizer.step()                                                ║
║  └── lr_scheduler.step()                                            ║
║                                                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 14.2 推理数据流

```
╔══════════════════════════════════════════════════════════════════════╗
║                        推理数据流                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  输入                                                                ║
║  └── image:          (H, W, 3)             PIL Image               ║
║                                                                     ║
║  预处理                                                              ║
║  ├── resize:         (768, 768, 3)         bilinear                ║
║  ├── normalize:      (3, 768, 768)         → [-1, 1]              ║
║  └── rgb_in:         (1, 3, 768, 768)      batch dim               ║
║                                                                     ║
║  编码                                                                ║
║  └── rgb_latent:     (1, 4, 96, 96)        VAE.encode              ║
║                                                                     ║
║  初始化                                                              ║
║  └── depth_latent:   (1, 4, 96, 96)        ~ N(0, 1)              ║
║                                                                     ║
║  去噪循环 (4步 DDIM, 重复5次集成)                                    ║
║  ╔═══════════════════════════════════════════════════════════╗       ║
║  ║  t = 999 → 749 → 499 → 249                              ║       ║
║  ║                                                          ║       ║
║  ║  cat:           (1, 8, 96, 96)     = [rgb, depth]       ║       ║
║  ║  feat:          (1, 320, 96, 96)   = conv_in(cat)       ║       ║
║  ║  slots:         (1, 16, 1024)      = aggregator(feat)   ║       ║
║  ║  slots_pad:     (1, 77, 1024)      = pad(slots)         ║       ║
║  ║  noise_pred:    (1, 4, 96, 96)     = unet(cat, t, slots)║       ║
║  ║  depth_latent:  (1, 4, 96, 96)     = scheduler.step()   ║       ║
║  ╚═══════════════════════════════════════════════════════════╝       ║
║                                                                     ║
║  解码                                                                ║
║  ├── stacked:        (1, 3, 768, 768)      VAE.decode              ║
║  ├── depth_mean:     (1, 1, 768, 768)      channel average         ║
║  └── depth:          (1, 1, 768, 768)      clip [0, 1]             ║
║                                                                     ║
║  后处理                                                              ║
║  ├── 集成:           (768, 768)            5次取中位数               ║
║  ├── resize:         (H, W)               回到原始分辨率             ║
║  └── colorize:       (H, W, 3)            Spectral 颜色映射        ║
║                                                                     ║
║  输出                                                                ║
║  ├── depth_np:       (H, W)               numpy, ∈ [0, 1]          ║
║  └── depth_colored:  PIL.Image             可视化                    ║
║                                                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 14.3 UNet 内部数据流

```
UNet2DConditionModel 内部结构:

输入: sample (B, 8, h, w), timestep, encoder_hidden_states (B, 77, 1024)

┌──────────────────────────────────────────────────────┐
│ conv_in: Conv2d(8 → 320, k=3, p=1)                  │
│ → (B, 320, h, w)                                     │
├──────────────────────────────────────────────────────┤
│ time_embedding: timestep → (B, 1280)                 │
├──────────────────────────────────────────────────────┤
│                                                      │
│ ═══ Encoder (下采样) ═══                              │
│                                                      │
│ down_block_0: CrossAttnDownBlock2D                   │
│   ├── ResBlock(320→320) × 2                          │
│   ├── CrossAttn(320, context=1024) × 2  ★ slots 在此 │
│   └── Downsample(320→320, stride=2)                  │
│   → (B, 320, h/2, w/2)                              │
│                                                      │
│ down_block_1: CrossAttnDownBlock2D                   │
│   ├── ResBlock(320→640) × 2                          │
│   ├── CrossAttn(640, context=1024) × 2  ★ slots 在此 │
│   └── Downsample(640→640, stride=2)                  │
│   → (B, 640, h/4, w/4)                              │
│                                                      │
│ down_block_2: CrossAttnDownBlock2D                   │
│   ├── ResBlock(640→1280) × 2                         │
│   ├── CrossAttn(1280, context=1024) × 2 ★ slots 在此 │
│   └── Downsample(1280→1280, stride=2)                │
│   → (B, 1280, h/8, w/8)                             │
│                                                      │
│ down_block_3: DownBlock2D                            │
│   └── ResBlock(1280→1280) × 2                        │
│   → (B, 1280, h/8, w/8)                             │
│                                                      │
│ ═══ 瓶颈 ═══                                         │
│                                                      │
│ mid_block: UNetMidBlock2DCrossAttn                   │
│   ├── ResBlock(1280→1280)                            │
│   ├── CrossAttn(1280, context=1024)     ★ slots 在此 │
│   └── ResBlock(1280→1280)                            │
│   → (B, 1280, h/8, w/8)                             │
│                                                      │
│ ═══ Decoder (上采样) ═══                              │
│                                                      │
│ up_block_0: UpBlock2D                                │
│   ├── ResBlock(2560→1280) × 3  (+ skip connection)   │
│   └── Upsample(1280→1280, stride=2)                  │
│   → (B, 1280, h/4, w/4)                             │
│                                                      │
│ up_block_1: CrossAttnUpBlock2D                       │
│   ├── ResBlock(1920→1280→640) × 3                    │
│   ├── CrossAttn(640→1280, context=1024) ★ slots 在此 │
│   └── Upsample(640→640, stride=2)                    │
│   → (B, 640, h/2, w/2)                              │
│                                                      │
│ up_block_2: CrossAttnUpBlock2D                       │
│   ├── ResBlock(960→640→320) × 3                      │
│   ├── CrossAttn(320→640, context=1024)  ★ slots 在此 │
│   └── Upsample(320→320, stride=2)                    │
│   → (B, 320, h, w)                                   │
│                                                      │
│ up_block_3: CrossAttnUpBlock2D                       │
│   ├── ResBlock(640→320) × 3                          │
│   └── CrossAttn(320, context=1024)      ★ slots 在此 │
│   → (B, 320, h, w)                                   │
│                                                      │
├──────────────────────────────────────────────────────┤
│ conv_norm_out: GroupNorm(32, 320)                     │
│ conv_act: SiLU                                       │
│ conv_out: Conv2d(320 → 4, k=3, p=1)                 │
│ → (B, 4, h, w)                                      │
└──────────────────────────────────────────────────────┘

★ CrossAttention 在每个标记的位置:
  Q = spatial_features  (来自 UNet 空间特征)
  K, V = slots          (来自 Slot Attention 输出)
  
  attn = softmax(Q @ K^T / sqrt(d)) @ V
  → 空间特征通过交叉注意力"查询" slots 中的语义信息
```

---

## 附录

### A. 关键超参数参考

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_slots` | 16 | 场景中的对象槽数量 |
| `slot_dim` | 1024 | Slot 嵌入维度 (SD v2) |
| `slot_iter` | 3 | Slot Attention 迭代次数 |
| `slot_ffn_dim` | 2048 | FFN 隐层维度 |
| `slot_input_dim` | 320 | conv_in 输出通道数 |
| `lr` | 3e-5 | 学习率 |
| `batch_size` | 2 | 每GPU batch size |
| `grad_accum` | 4 | 梯度累积步数 |
| `max_iter` | 5000 | 总训练步数 |
| `denoising_steps` | 4 | 推理去噪步数 |
| `ensemble_size` | 5 | 推理集成次数 |
| `processing_res` | 768 | 推理处理分辨率 |
| `vae_scale` | 0.18215 | VAE 缩放因子 |

### B. 文件引用索引

| 组件 | 文件路径 | 关键行号 |
|------|----------|----------|
| SlotInitializer | `src/model/slot_attention.py` | ~L20-80 |
| SlotAttention | `src/model/slot_attention.py` | ~L80-200 |
| SlotAggregator | `src/model/slot_attention.py` | ~L200-300 |
| Pipeline 初始化 | `marigold/marigold_depth_pipeline.py` | L126-175 |
| Pipeline 推理 | `marigold/marigold_depth_pipeline.py` | L427-507 |
| Pipeline 编码 | `marigold/marigold_depth_pipeline.py` | L509-526 |
| Pipeline 解码 | `marigold/marigold_depth_pipeline.py` | L528-546 |
| Trainer 初始化 | `src/trainer/marigold_depth_trainer.py` | L61-150 |
| conv_in 修改 | `src/trainer/marigold_depth_trainer.py` | L191-210 |
| 训练步骤 | `src/trainer/marigold_depth_trainer.py` | L212-420 |
| Slot 聚合 | `src/trainer/marigold_depth_trainer.py` | L306-321 |
| UNet 前向 | `src/trainer/marigold_depth_trainer.py` | L324-326 |
| 损失计算 | `src/trainer/marigold_depth_trainer.py` | L343-351 |
| 基础数据集 | `src/dataset/base_depth_dataset.py` | 全文件 |
| 混合采样 | `src/dataset/mixed_sampler.py` | 全文件 |
| 损失函数 | `src/util/loss.py` | 全文件 |
| 评估指标 | `src/util/metric.py` | 全文件 |

### C. 常用命令

```bash
# 训练
python script/depth/train.py \
    --config config/train_marigold_depth.yaml \
    --base_data_dir /path/to/data

# 推理
python script/depth/run.py \
    --checkpoint /path/to/checkpoint \
    --input_rgb_dir /path/to/images \
    --output_dir /path/to/output

# 评估
python script/depth/eval.py \
    --checkpoint /path/to/checkpoint \
    --dataset nyu_v2 \
    --base_data_dir /path/to/data
```
