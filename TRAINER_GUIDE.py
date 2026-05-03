"""
示例：如何修改 marigold_depth_trainer.py 支持 Slot Attention 训练

这个文件展示了需要在 MarigoldDepthTrainer 中做的关键修改
"""

# ============================================================================
# 修改 1: 修改 __init__ 方法 - 设置 Slot 模块的可训练性
# ============================================================================

def __init__(
    self,
    cfg: OmegaConf,
    model: MarigoldDepthPipeline,
    train_dataloader: DataLoader,
    device,
    out_dir_ckpt,
    out_dir_eval,
    out_dir_vis,
    accumulation_steps: int,
    val_dataloaders: List[DataLoader] = None,
    vis_dataloaders: List[DataLoader] = None,
):
    # ... 原有初始化代码 ...

    # ========== 修改：Trainability ==========
    self.model.vae.requires_grad_(False)
    self.model.text_encoder.requires_grad_(False)
    self.model.unet.requires_grad_(True)

    # 新增：设置 Slot 模块可训练
    if self.model.use_slot_attention and self.model.slot_aggregator is not None:
        self.model.slot_aggregator.requires_grad_(True)
        logging.info("Slot Aggregator is trainable")

    # ========== 修改：Optimizer - 包含 Slot 参数 ==========
    lr = self.cfg.lr

    # 收集所有可训练参数
    trainable_params = []

    # UNet 参数
    unet_params = {
        'params': self.model.unet.parameters(),
        'lr': lr,
        'name': 'unet'
    }
    trainable_params.append(unet_params)

    # Slot 参数（如果启用）
    if self.model.use_slot_attention and self.model.slot_aggregator is not None:
        slot_params = {
            'params': self.model.slot_aggregator.parameters(),
            'lr': lr,  # 初始使用相同学习率
            'name': 'slot'
        }
        trainable_params.append(slot_params)
        logging.info(f"Added Slot Aggregator parameters to optimizer")

    self.optimizer = Adam(trainable_params, lr=lr)

    # ... 原有的 LR scheduler 等代码 ...

    # ========== 新增：Slot 训练策略配置 ==========
    if self.model.use_slot_attention:
        # 从配置读取 warmup 策略
        slot_cfg = self.cfg.get('slot_attention', {})
        training_cfg = slot_cfg.get('training', {})

        self.warmup_slot_only_iters = training_cfg.get('warmup_slot_only_iters', 500)
        self.warmup_small_lr_iters = training_cfg.get('warmup_small_lr_iters', 1000)
        self.unet_lr_ratio = training_cfg.get('unet_lr_ratio', 0.1)

        self.use_slot_regularization = training_cfg.get('use_slot_regularization', True)
        self.slot_reg_weight = training_cfg.get('slot_reg_weight', 0.01)
        self.slot_grad_clip = training_cfg.get('slot_grad_clip', 1.0)

        logging.info(
            f"Slot training strategy: "
            f"warmup_slot_only={self.warmup_slot_only_iters}, "
            f"warmup_small_lr={self.warmup_small_lr_iters}, "
            f"unet_lr_ratio={self.unet_lr_ratio}"
        )


# ============================================================================
# 修改 2: 添加 Slot 正则化损失计算方法
# ============================================================================

def compute_slot_regularization(self, slots, reference_embed=None):
    """
    计算 slot 正则化损失，防止 slots 分布偏离 text embedding 太远

    Args:
        slots: Slot embeddings, shape (B, num_slots, slot_dim)
        reference_embed: Reference embedding (e.g., empty_text_embed), shape (B, 77, 768)

    Returns:
        reg_loss: Regularization loss (scalar)
    """
    if reference_embed is None:
        reference_embed = self.empty_text_embed

    # 只使用前 num_slots 个 reference embeddings
    num_slots = slots.shape[1]
    ref = reference_embed[:, :num_slots, :]  # (B, num_slots, slot_dim)

    # 1. 均值匹配
    slot_mean = slots.mean(dim=1)  # (B, slot_dim)
    ref_mean = ref.mean(dim=1)  # (B, slot_dim)
    mean_loss = F.mse_loss(slot_mean, ref_mean)

    # 2. 标准差匹配
    slot_std = slots.std(dim=1)  # (B, slot_dim)
    ref_std = ref.std(dim=1)  # (B, slot_dim)
    std_loss = F.mse_loss(slot_std, ref_std)

    # 3. 范数约束（防止 slots 过大）
    norm_loss = (slots.norm(dim=-1).mean() - ref.norm(dim=-1).mean()).abs()

    # 总正则化损失
    reg_loss = mean_loss + std_loss + 0.1 * norm_loss

    return reg_loss


# ============================================================================
# 修改 3: 修改训练阶段控制（渐进式训练）
# ============================================================================

def adjust_training_stage(self):
    """
    根据当前 iteration 调整训练阶段

    阶段 1 (0 - warmup_slot_only_iters): 只训练 Slot，冻结 UNet
    阶段 2 (warmup_slot_only_iters - warmup_small_lr_iters): 小学习率训练 UNet + Slot
    阶段 3 (warmup_small_lr_iters+): 正常训练
    """
    if not self.model.use_slot_attention:
        return

    current_iter = self.effective_iter

    # 阶段 1: 只训练 Slot
    if current_iter < self.warmup_slot_only_iters:
        if current_iter == 0:
            logging.info(
                f"[Stage 1] Training only Slot Aggregator "
                f"(iters 0-{self.warmup_slot_only_iters})"
            )
        self.model.unet.requires_grad_(False)
        self.model.slot_aggregator.requires_grad_(True)

    # 阶段 2: 小学习率训练 UNet
    elif current_iter < self.warmup_small_lr_iters:
        if current_iter == self.warmup_slot_only_iters:
            logging.info(
                f"[Stage 2] Training UNet with small LR "
                f"(iters {self.warmup_slot_only_iters}-{self.warmup_small_lr_iters})"
            )
            # 解冻 UNet
            self.model.unet.requires_grad_(True)

            # 调整 UNet 学习率
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'unet':
                    param_group['lr'] = self.cfg.lr * self.unet_lr_ratio
                    logging.info(f"UNet LR set to {param_group['lr']}")

    # 阶段 3: 正常训练
    else:
        if current_iter == self.warmup_small_lr_iters:
            logging.info(
                f"[Stage 3] Normal training "
                f"(iters {self.warmup_small_lr_iters}+)"
            )
            # 恢复 UNet 学习率
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'unet':
                    param_group['lr'] = self.cfg.lr
                    logging.info(f"UNet LR restored to {param_group['lr']}")


# ============================================================================
# 修改 4: 修改 train_step 方法 - 集成 Slot 训练逻辑
# ============================================================================

def train_step(self, batch):
    """
    单步训练
    """
    # ========== 新增：调整训练阶段 ==========
    if self.model.use_slot_attention:
        self.adjust_training_stage()

    # ... 原有的数据准备代码 ...
    rgb = batch["rgb_norm"].to(self.device)
    depth_gt_for_latent = batch[self.gt_depth_type].to(self.device)
    valid_mask_for_latent = batch[self.gt_mask_type].to(self.device)

    # 编码
    rgb_latent = self.model.encode_rgb(rgb)
    gt_depth_latent = self.model.encode_depth(depth_gt_for_latent)

    # ========== 修改：获取 encoder_hidden_states ==========
    if self.model.use_slot_attention:
        # 使用 slot aggregation
        encoder_hidden_states, slot_attn = self.model.aggregate_slots(rgb_latent)
        # encoder_hidden_states: (B, 77, 768)
        # slot_attn: (B, num_slots, H*W) - 用于可视化
    else:
        # 使用原始 empty text embedding
        encoder_hidden_states = self.empty_text_embed
        slot_attn = None

    # ... 原有的噪声添加代码 ...
    # 添加噪声
    noise = torch.randn_like(gt_depth_latent)
    if self.apply_multi_res_noise:
        noise = multi_res_noise_like(...)

    timesteps = torch.randint(...)
    noisy_depth_latent = self.training_noise_scheduler.add_noise(...)

    # UNet 前向传播
    unet_input = torch.cat([rgb_latent, noisy_depth_latent], dim=1)
    noise_pred = self.model.unet(
        unet_input,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,  # 使用 slots
        return_dict=False,
    )[0]

    # 计算主损失
    if "epsilon" == self.prediction_type:
        target = noise
    elif "v_prediction" == self.prediction_type:
        target = self.training_noise_scheduler.get_velocity(...)

    loss = self.loss(noise_pred, target)

    # ========== 新增：添加 Slot 正则化损失 ==========
    if self.model.use_slot_attention and self.use_slot_regularization:
        # 提取实际的 slots (去掉 padding)
        num_slots = self.model.slot_aggregator.num_slots
        slots = encoder_hidden_states[:, :num_slots, :]

        # 计算正则化损失
        reg_loss = self.compute_slot_regularization(slots, self.empty_text_embed)
        loss = loss + self.slot_reg_weight * reg_loss

        # 记录到 tensorboard
        tb_logger.log_dic(
            {
                "train/slot_reg_loss": reg_loss.item(),
                "train/main_loss": (loss - self.slot_reg_weight * reg_loss).item(),
            },
            global_step=self.effective_iter,
        )

    # 反向传播
    loss.backward()

    # ========== 修改：梯度裁剪（分别处理 UNet 和 Slot）==========
    accumulated_step = (self.n_batch_in_epoch + 1) % self.gradient_accumulation_steps

    if accumulated_step == 0 or (self.n_batch_in_epoch + 1) == len(self.train_loader):
        # UNet 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.unet.parameters(),
            max_norm=1.0
        )

        # Slot 梯度裁剪（如果启用）
        if self.model.use_slot_attention and self.model.slot_aggregator is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.slot_aggregator.parameters(),
                max_norm=self.slot_grad_clip
            )

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.effective_iter += 1

    # ... 原有的 metrics 记录代码 ...

    return loss.item()


# ============================================================================
# 修改 5: (可选) 添加 Slot Attention 可视化
# ============================================================================

def visualize_slot_attention(self, batch, slot_attn, save_dir, step):
    """
    可视化 slot attention maps

    Args:
        batch: 输入 batch
        slot_attn: Attention maps, shape (B, num_slots, H*W)
        save_dir: 保存目录
        step: 当前 step
    """
    if slot_attn is None:
        return

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    B, num_slots, HW = slot_attn.shape
    H = W = int(np.sqrt(HW))

    # 重塑 attention maps
    attn_maps = slot_attn.reshape(B, num_slots, H, W)  # (B, num_slots, H, W)

    # 只可视化第一个样本
    rgb = batch["rgb_norm"][0].cpu()  # (3, H, W)
    attn = attn_maps[0].cpu()  # (num_slots, H, W)

    # 创建图像
    fig, axes = plt.subplots(2, num_slots, figsize=(num_slots * 3, 6))

    # 第一行：RGB 图像叠加 attention
    for i in range(num_slots):
        ax = axes[0, i]
        ax.imshow(rgb.permute(1, 2, 0))
        ax.imshow(attn[i], alpha=0.5, cmap='jet')
        ax.set_title(f'Slot {i+1}')
        ax.axis('off')

    # 第二行：纯 attention maps
    for i in range(num_slots):
        ax = axes[1, i]
        ax.imshow(attn[i], cmap='viridis')
        ax.set_title(f'Attention {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'slot_attn_step_{step}.png'))
    plt.close()


# ============================================================================
# 总结：需要修改的位置
# ============================================================================
"""
1. __init__ 方法：
   - 设置 slot_aggregator 可训练
   - 将 slot 参数添加到 optimizer
   - 读取 slot 训练策略配置

2. 添加新方法：
   - compute_slot_regularization(): 计算正则化损失
   - adjust_training_stage(): 渐进式训练阶段控制
   - visualize_slot_attention(): 可视化 attention maps

3. 修改 train_step 方法：
   - 调用 adjust_training_stage()
   - 使用 aggregate_slots() 获取 encoder_hidden_states
   - 添加 slot 正则化损失
   - 分别裁剪 UNet 和 Slot 的梯度

4. (可选) 修改 validation/visualization 方法：
   - 在验证时也使用 slot attention
   - 可视化 slot attention maps

关键点：
- 渐进式训练防止破坏 SD 先验
- 正则化损失保持 slots 分布合理
- 分别管理 UNet 和 Slot 的学习率和梯度
- 可视化 attention 帮助调试
"""
