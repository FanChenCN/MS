"""
示例：如何修改 marigold_depth_pipeline.py 集成 Slot Attention

这个文件展示了需要在 MarigoldDepthPipeline 中做的关键修改
"""

# ============================================================================
# 修改 1: 在文件开头添加 import
# ============================================================================
# 在 marigold_depth_pipeline.py 的 import 部分添加：

from src.model.slot_attention import SlotAggregator


# ============================================================================
# 修改 2: 修改 MarigoldDepthPipeline.__init__()
# ============================================================================
# 在 MarigoldDepthPipeline 类的 __init__ 方法中添加 slot 参数：

def __init__(
    self,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: Union[DDIMScheduler, LCMScheduler],
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scale_invariant: Optional[bool] = True,
    shift_invariant: Optional[bool] = True,
    default_denoising_steps: Optional[int] = None,
    default_processing_resolution: Optional[int] = None,
    # ========== 新增 Slot 参数 ==========
    use_slot_attention: Optional[bool] = False,
    num_slots: Optional[int] = 7,
    slot_dim: Optional[int] = 768,
    slot_num_iter: Optional[int] = 3,
    slot_ffn_dim: Optional[int] = 2048,
    slot_input_dim: Optional[int] = 320,
):
    super().__init__()

    # 原有的 register_modules
    self.register_modules(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # 原有的 register_to_config
    self.register_to_config(
        scale_invariant=scale_invariant,
        shift_invariant=shift_invariant,
        default_denoising_steps=default_denoising_steps,
        default_processing_resolution=default_processing_resolution,
        # ========== 新增：注册 slot 配置 ==========
        use_slot_attention=use_slot_attention,
        num_slots=num_slots,
        slot_dim=slot_dim,
        slot_num_iter=slot_num_iter,
        slot_ffn_dim=slot_ffn_dim,
        slot_input_dim=slot_input_dim,
    )

    # ========== 新增：初始化 Slot Aggregator ==========
    self.use_slot_attention = use_slot_attention
    if use_slot_attention:
        self.slot_aggregator = SlotAggregator(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_input_dim,
            num_iter=slot_num_iter,
            ffn_dim=slot_ffn_dim,
            dropout=0.0,
        )
        # 注册为模块，这样会被自动保存/加载
        self.register_modules(slot_aggregator=self.slot_aggregator)
    else:
        self.slot_aggregator = None

    # 原有的其他初始化代码...
    self.empty_text_embed = None


# ============================================================================
# 修改 3: 添加 aggregate_slots 方法
# ============================================================================
# 在 MarigoldDepthPipeline 类中添加新方法：

def aggregate_slots(self, rgb_latent):
    """
    从 RGB latent 中聚合 object slots

    Args:
        rgb_latent: RGB image latent from VAE, shape (B, 4, H, W)

    Returns:
        slots: Object slots, shape (B, 77, 768) - padded to match text_embed
        attn: Attention maps, shape (B, num_slots, H*W) or None
    """
    if not self.use_slot_attention or self.slot_aggregator is None:
        # 如果不使用 slot，返回原始的 empty text embedding
        return self.empty_text_embed, None

    # 1. 通过 UNet 的第一层提取特征
    # 注意：这里使用 no_grad 避免影响 VAE 的梯度
    with torch.no_grad():
        # 使用 UNet 的 conv_in 提取特征
        feat = self.unet.conv_in(rgb_latent)  # (B, 320, H, W)

    # 2. 聚合 slots
    slots, attn = self.slot_aggregator(feat)  # (B, num_slots, slot_dim)

    # 3. 格式化 slots 以匹配 text embedding 的形状
    # SD v2 期望 (B, 77, 768)
    B, num_slots, slot_dim = slots.shape

    if num_slots < 77:
        # Pad slots to length 77
        padding = torch.zeros(
            B, 77 - num_slots, slot_dim,
            device=slots.device,
            dtype=slots.dtype
        )
        slots_padded = torch.cat([slots, padding], dim=1)  # (B, 77, slot_dim)
    elif num_slots > 77:
        # Truncate (不推荐，应该设置 num_slots <= 77)
        slots_padded = slots[:, :77, :]
    else:
        slots_padded = slots

    return slots_padded, attn


# ============================================================================
# 修改 4: 修改 single_infer 方法（训练时的 forward）
# ============================================================================
# 找到 single_infer 方法，修改如下：

def single_infer(
    self,
    rgb_in: torch.Tensor,
    num_inference_steps: int,
    show_pbar: bool,
    generator: Union[torch.Generator, None],
) -> torch.Tensor:
    """
    单次推理（训练时也会调用这个方法）
    """
    device = rgb_in.device

    # 1. 编码 RGB 图像
    rgb_latent = self.encode_rgb(rgb_in)  # (B, 4, H, W)

    # ========== 修改：使用 slot aggregation 替代 text embedding ==========
    if self.use_slot_attention:
        # 使用 slot attention 聚合特征
        encoder_hidden_states, slot_attn = self.aggregate_slots(rgb_latent)
        # encoder_hidden_states: (B, 77, 768)
    else:
        # 使用原始的 empty text embedding
        encoder_hidden_states = self.empty_text_embed.to(device)
        slot_attn = None

    # 2. 初始化噪声
    depth_latent = torch.randn(
        rgb_latent.shape,
        device=device,
        dtype=rgb_latent.dtype,
        generator=generator,
    )  # (B, 4, H, W)

    # 3. 设置 scheduler
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 4. 去噪循环
    for t in tqdm(timesteps, desc="Denoising", disable=not show_pbar):
        # 拼接 RGB latent 和 depth latent
        unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # (B, 8, H, W)

        # UNet 预测噪声
        noise_pred = self.unet(
            unet_input,
            t,
            encoder_hidden_states=encoder_hidden_states,  # 使用 slots
            return_dict=False,
        )[0]

        # Scheduler step
        depth_latent = self.scheduler.step(
            noise_pred, t, depth_latent, generator=generator
        ).prev_sample

    # 5. 解码 depth latent
    depth = self.decode_depth(depth_latent)

    # 返回 depth 和 attention（用于可视化）
    return depth, slot_attn


# ============================================================================
# 修改 5: 修改 __call__ 方法（如果需要返回 attention）
# ============================================================================
# 如果想在推理时也返回 attention maps，可以修改 __call__ 方法
# 这是可选的，主要用于可视化

def __call__(
    self,
    rgb_in: torch.Tensor,
    # ... 其他参数 ...
    return_attention: bool = False,  # 新增参数
) -> MarigoldDepthOutput:
    """
    主推理接口
    """
    # ... 原有代码 ...

    # 调用 single_infer
    depth_pred, slot_attn = self.single_infer(...)

    # ... 原有的后处理代码 ...

    # 如果需要返回 attention
    if return_attention and slot_attn is not None:
        return MarigoldDepthOutput(
            depth_np=depth_np,
            depth_colored=depth_colored,
            uncertainty=uncertainty,
            slot_attention=slot_attn,  # 需要修改 MarigoldDepthOutput 类
        )
    else:
        return MarigoldDepthOutput(
            depth_np=depth_np,
            depth_colored=depth_colored,
            uncertainty=uncertainty,
        )


# ============================================================================
# 总结：需要修改的位置
# ============================================================================
"""
1. Import 部分：添加 SlotAggregator
2. __init__ 方法：
   - 添加 slot 相关参数
   - 初始化 slot_aggregator
   - 注册到 config 和 modules
3. 添加 aggregate_slots 方法
4. 修改 single_infer 方法：
   - 调用 aggregate_slots 获取 slots
   - 将 slots 传递给 UNet
5. (可选) 修改 __call__ 和 MarigoldDepthOutput 以返回 attention

关键点：
- slot_aggregator 需要注册为 module，这样会自动保存/加载
- aggregate_slots 中使用 no_grad 提取特征，避免影响 VAE
- slots 需要 pad 到 77 以匹配 SD v2 的 text embedding 长度
- 返回的 attention 可用于可视化 slot 分割结果
"""
