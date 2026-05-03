# 快速启动指南

## 环境准备

```bash
cd /home/chenfan/projectsVScode/CMarigold/Marigold_Slots

# 激活虚拟环境
source venv/marigold/bin/activate

# 确认 wandb 已登录
wandb login --relogin  # 如果需要重新登录
```

## 实验一：16 Slots

```bash
export BASE_DATA_DIR=/path/to/your/data
export BASE_CKPT_DIR=/path/to/your/checkpoints

python script/depth/train.py \
  --config config/train_marigold_depth.yaml \
  --add_datetime_prefix
```

**配置**：
- num_slots: 16
- slot_input_dim: 516 (512 VAE mid + 4 rgb_latent)
- 输出: (16, 1024) → pad → (77, 1024)

## 实验二：77 Slots

```bash
export BASE_DATA_DIR=/path/to/your/data
export BASE_CKPT_DIR=/path/to/your/checkpoints

python script/depth/train.py \
  --config config/train_slots77.yaml \
  --add_datetime_prefix
```

**配置**：
- num_slots: 77
- slot_input_dim: 516 (512 VAE mid + 4 rgb_latent)
- 输出: (77, 1024) 直接对齐

## 查看训练日志

### 本地 TensorBoard
```bash
tensorboard --logdir output/
```

### WandB 在线查看
1. 打开浏览器访问: https://wandb.ai
2. 进入项目: `marigold-slots-vae-mid`
3. 查看两个实验的对比曲线

## 恢复训练

```bash
python script/depth/train.py \
  --resume_run output/<job_name>/checkpoint/latest
```

## 调试模式（快速验证）

```bash
# 使用小数据集快速测试
python script/depth/train.py \
  --config config/train_debug_depth.yaml \
  --add_datetime_prefix
```

## 关键参数说明

### 数据相关
- `BASE_DATA_DIR`: 数据集根目录
- `BASE_CKPT_DIR`: 预训练模型目录（需要包含 SD v2 checkpoint）

### 训练相关
- `--config`: 配置文件路径
- `--resume_run`: 恢复训练的 checkpoint 路径
- `--add_datetime_prefix`: 在输出目录名前添加时间戳
- `--no_wandb`: 禁用 wandb（不推荐）
- `--exit_after`: 训练 X 分钟后自动保存并退出

### 输出目录结构
```
output/<job_name>/
├── checkpoint/          # 模型 checkpoint
│   ├── latest/         # 最新 checkpoint（可恢复训练）
│   └── iter_XXXXXX/    # 定期备份的 checkpoint
├── tensorboard/        # TensorBoard 日志
├── evaluation/         # 验证集评估结果
├── visualization/      # 可视化结果
├── config.yaml         # 训练配置快照
└── logging.log         # 训练日志
```

## 监控训练

### 关键指标
- `train/loss`: 训练损失（应该逐渐下降）
- `val/abs_relative_difference`: 验证集相对误差（越小越好）
- `val/delta1_acc`: Delta1 准确率（越大越好）
- `lr`: 学习率（warmup + exponential decay）

### 正常训练特征
- Loss 在前 100 步快速下降
- Warmup 阶段（前 100 步）学习率逐渐上升
- 之后学习率指数衰减
- 验证指标在 250 步后开始有意义

## 常见问题

### 1. CUDA Out of Memory
```bash
# 减小 batch size
# 修改 config/*.yaml 中的:
dataloader:
  max_train_batch_size: 1  # 从 2 改为 1
```

### 2. WandB 未记录数据
```bash
# 检查登录状态
wandb login --relogin

# 检查网络连接
ping wandb.ai
```

### 3. 找不到数据集
```bash
# 确认环境变量设置
echo $BASE_DATA_DIR
echo $BASE_CKPT_DIR

# 检查数据集路径
ls $BASE_DATA_DIR
```

### 4. 维度不匹配错误
```bash
# 确认配置文件中 slot_input_dim = 516
grep slot_input_dim config/train_marigold_depth.yaml
```

## 实验对比

### 在 WandB 中对比两个实验

1. 打开 WandB 项目页面
2. 选择 Workspace
3. 在左侧勾选两个实验的 run
4. 点击 "Add panel" → "Line plot"
5. 选择要对比的指标（如 `train/loss`）
6. 两条曲线会自动叠加显示

### 导出对比结果

```python
import wandb

api = wandb.Api()
runs = api.runs("your-entity/marigold-slots-vae-mid")

for run in runs:
    print(f"{run.name}: {run.summary}")
```

## 下一步

1. 等待两个实验训练完成（约 5000 iterations）
2. 在 WandB 中对比关键指标
3. 分析 slot attention map（如果实现了可视化）
4. 根据结果调整超参数或尝试其他 slot 数量

## 参考资料

- Marigold 论文: https://arxiv.org/abs/2312.02145
- Slot Attention 论文: https://arxiv.org/abs/2006.15055
- WandB 文档: https://docs.wandb.ai
- 修改详情: 查看 `MODIFICATIONS.md`
