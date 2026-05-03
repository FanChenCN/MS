# Marigold_Slots 修改总结

## 修改日期
2026-04-28

## 修改目标
将 Slot Attention 的输入从 UNet conv_in 特征 (320维) 改为 VAE encoder 中间特征 (512维) + RGB latent (4维)，以提供更丰富的语义信息用于 object-centric learning。

---

## 架构变化

### 修改前
```
rgb(B,3,256,256) → VAE.encode → rgb_latent(B,4,32,32)
                                      ↓
                              cat_latents(B,8,32,32)
                                ↓                ↓
                        unet.conv_in          unet(cat_latents, t, slots)
                        feat(B,320,32,32)
                                ↓
                        aggregator(feat)
                        slots(B,16,1024) → pad → (B,77,1024)
```

### 修改后
```
rgb(B,3,256,256) → VAE.encoder → rgb_latent(B,4,32,32)
                       ↓ hook
                vae_mid_feat(B,512,32,32)
                       ↓
        cat(vae_mid_feat, rgb_latent) → (B,516,32,32)
                       ↓
                aggregator(input_dim=516)
                slots(B,16/77,1024) → pad(if needed) → (B,77,1024)
                       ↓
depth_gt → VAE.encode + noise → noisy_latent(B,4,32,32)
                                      ↓
                          cat_latents(B,8,32,32) → unet(cat_latents, t, slots)
                                                        ↓
                                                  model_pred(B,4,32,32)
```

---

## 文件修改清单

### 1. `src/model/slot_attention.py`
**修改内容**：
- `SlotAggregator.__init__()` 的 `input_dim` 默认值从 320 改为 516

**代码位置**：第 232 行

### 2. `marigold/marigold_depth_pipeline.py`
**修改内容**：
- 添加 `_register_vae_hook()` 方法，在 VAE encoder 的 `down_blocks[2]` 注册 forward hook
- 修改 `__init__()` 调用 hook 注册，添加 `self.vae_mid_features` 存储中间特征
- 修改 `slot_input_dim` 默认值为 516
- 修改 `encode_rgb()` 文档说明中间特征通过 hook 存储
- 修改 `single_infer()` 方法：
  - 提取 VAE 中间特征
  - 拼接中间特征和 rgb_latent 作为 slot 输入
  - 移除 `batch_empty_text_embed`，直接使用 slots

**代码位置**：第 123-193 行, 457-490 行, 509-527 行

### 3. `src/trainer/marigold_depth_trainer.py`
**修改内容**：
- 修改 `training_step()` 方法：
  - 在 `encode_rgb()` 后提取 `vae_mid_features`
  - 拼接 VAE 中间特征和 rgb_latent
  - 移除 `unet.conv_in` 特征提取
  - 使用拼接后的特征进行 slot 聚合

**代码位置**：第 257-311 行

### 4. `config/train_marigold_depth.yaml`
**修改内容**：
- `pipeline.kwargs.slot_input_dim`: 320 → 516

**代码位置**：第 20 行

### 5. `config/wandb.yaml`
**修改内容**：
- `wandb.project`: "marigold" → "marigold-slots-vae-mid"

**代码位置**：第 3 行

### 6. `script/depth/train.py`
**修改内容**：
- 移除 wandb 配置中的 `"mode": "disabled"`，启用 wandb 日志

**代码位置**：第 196-199 行

### 7. `config/train_slots77.yaml` (新建)
**内容**：
- 复制自 `train_marigold_depth.yaml`
- `num_slots`: 16 → 77（实验二配置）

---

## 实验配置

### 实验一：16 slots (需要 padding)
```bash
python script/depth/train.py --config config/train_marigold_depth.yaml
```

**配置参数**：
- `num_slots`: 16
- `slot_dim`: 1024
- `slot_input_dim`: 516
- Slots 输出 (16, 1024) → pad → (77, 1024)

### 实验二：77 slots (直接对齐)
```bash
python script/depth/train.py --config config/train_slots77.yaml
```

**配置参数**：
- `num_slots`: 77
- `slot_dim`: 1024
- `slot_input_dim`: 516
- Slots 输出 (77, 1024)，无需 padding

---

## WandB 使用指南

### 1. 登录（已完成）
```bash
wandb login
```

### 2. 启动训练
训练会自动记录到 wandb 项目 `marigold-slots-vae-mid`

### 3. 查看实验对比
1. 打开 https://wandb.ai
2. 进入项目 `marigold-slots-vae-mid`
3. 在 Workspace 中查看两个实验的曲线对比

### 4. 关键指标
- `train/loss`: 训练损失
- `val/abs_relative_difference`: 验证集相对误差
- `val/delta1_acc`: Delta1 准确率
- `lr`: 学习率变化

### 5. 对比功能
- **Parallel Coordinates**: 超参数与指标关系
- **Table View**: 最终指标对比表格
- **Custom Charts**: 自定义对比图表
- **Group by**: 按 `num_slots` 分组对比

---

## 技术细节

### VAE Encoder Hook 位置
- **Hook 点**: `vae.encoder.down_blocks[2]`
- **输出维度**: (B, 512, 32, 32)
- **原因**: 这是 VAE encoder 最后一个下采样块的输出，包含丰富的高维语义特征

### Slot 输入维度计算
```
slot_input_dim = 512 (VAE mid features) + 4 (rgb_latent) = 516
```

### Slot 输出对齐
- **SD v2 text embedding**: (77, 1024)
  - 77: CLIP tokenizer max_length
  - 1024: OpenCLIP ViT-H embedding dimension
- **实验一**: (16, 1024) → zero padding → (77, 1024)
- **实验二**: (77, 1024) → 直接对齐

### UNet Cross-Attention
- **输入**: `encoder_hidden_states` = slots (B, 77, 1024)
- **替代**: 原本的 `empty_text_embed`
- **作用**: 提供 object-centric 的语义条件指导深度预测

---

## 验证修改

### 检查 VAE hook 是否工作
```python
# 在 training_step 中添加调试代码
print(f"vae_mid_feat shape: {vae_mid_feat.shape}")  # 应该是 (B, 512, 32, 32)
print(f"rgb_latent shape: {rgb_latent.shape}")      # 应该是 (B, 4, 32, 32)
print(f"slot_input shape: {slot_input.shape}")      # 应该是 (B, 516, 32, 32)
print(f"slots shape: {slots.shape}")                # 应该是 (B, 16/77, 1024)
```

### 检查维度对齐
```python
# 在 single_infer 中添加调试代码
assert slots.shape == (B, 77, 1024), f"Slots shape mismatch: {slots.shape}"
```

---

## 预期效果

### 理论优势
1. **更丰富的语义信息**: 512维 VAE 中间特征 >> 320维 conv_in 特征
2. **更好的 object-centric 表示**: 高维特征有助于 slot attention 分离对象
3. **更强的条件指导**: 语义丰富的 slots 提供更好的 cross-attention 条件

### 实验对比目标
- **实验一 vs 实验二**: 16 slots (sparse) vs 77 slots (dense)
- **关键指标**: 
  - 训练收敛速度
  - 验证集深度估计精度
  - Slot attention 的可解释性（通过 attention map 可视化）

---

## 注意事项

1. **内存占用**: VAE 中间特征 (512维) 比 conv_in 特征 (320维) 占用更多内存
2. **计算开销**: Slot attention 输入维度增加，计算量略有增加
3. **Hook 副作用**: 确保 hook 在每次 forward 时正确更新 `vae_mid_features`
4. **Gradient flow**: VAE encoder 在训练时是冻结的 (`requires_grad=False`)，中间特征不参与梯度回传

---

## 后续优化方向

1. **多尺度特征融合**: 尝试融合多个 VAE encoder 层的特征
2. **Slot 数量搜索**: 尝试其他 slot 数量（如 32, 49）
3. **Slot 正则化**: 添加 slot 的稀疏性或多样性正则化
4. **Attention 可视化**: 可视化 slot attention map 验证 object-centric 效果

---

## 问题排查

### 如果训练报错
1. **维度不匹配**: 检查 `slot_input_dim` 是否为 516
2. **Hook 未触发**: 检查 `vae_mid_features` 是否为 None
3. **内存溢出**: 减小 `batch_size` 或 `gradient_accumulation_steps`

### 如果 wandb 未记录
1. 检查是否登录: `wandb login`
2. 检查配置: `config/wandb.yaml` 中的 project 名称
3. 检查网络: wandb 需要联网上传数据

---

## 联系方式
如有问题，请查看：
- WandB 文档: https://docs.wandb.ai
- Marigold 项目: https://github.com/prs-eth/Marigold
- Slot Attention 论文: https://arxiv.org/abs/2006.15055
