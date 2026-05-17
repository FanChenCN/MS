import torch 
import timm
import torch.nn as nn
from einops import rearrange
import torchvision.transforms as transform
import torch.nn.functional as F
import os
class DINO2ViT(nn.Module):
    """
    https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8
    """

    def __init__(
        self,
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        in_size=518, # TODO 输入图像对齐
        rearrange_out=True,
        norm_out=True,
    ):
        super().__init__()
        # dict(
        #     patch_size=14,
        #     embed_dim=384,
        #     depth=12,
        #     num_heads=6,
        #     init_values=1e-05,
        #     reg_tokens=4,
        #     no_embed_class=True,
        #     pretrained_cfg="lvd142m",
        #     pretrained_cfg_overlay=None,
        #     cache_dir=None,
        # )
        os.environ.setdefault("HF_HUB_OFFLINE", "1") # 在线加载 0
        model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
        from safetensors.torch import load_file
        local_weight = "/home/chenfan/.cache/huggingface/hub/models--timm--vit_small_patch14_reg4_dinov2.lvd142m/snapshots/main/model.safetensors"
        state_dict = load_file(local_weight)
        model.load_state_dict(state_dict, strict=False)

        self.patch_size = 14
        self.in_size = in_size
        assert self.patch_size == 14

        self.cls_token = model.cls_token
        self.reg_token = model.reg_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = model.norm if norm_out else nn.Identity()

        for k, v in model.__dict__.items():
            if any(
                [
                    k.startswith("__") and k.endswith("__"),
                    k.startswith("_"),
                    isinstance(v, nn.Module),
                    isinstance(v, nn.Parameter),
                    hasattr(self, k),
                ]
            ):
                continue
            else:
                setattr(self, k, v)
        assert hasattr(self, "num_prefix_tokens")

        __class__._pos_embed = model.__class__._pos_embed
        __class__.forward_features = model.__class__.forward_features

        self.rearrange_out = rearrange_out
        self.register_buffer(
            "imgnet_mean",
            torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        )
        self.register_buffer(
            "imgnet_std",
            torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        )

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def forward(self, input):
        """
        input: shape=(b,c,h,w), in [-1,-1] float
        returns:(B,C,h,w),valid_mask(B,h,w)
        """
        # with pt.inference_mode(True):  # infer+compile: errors
        B,C,H_orig,W_orig = input.shape
        
        # 1.pad 到 14 的整数倍490x644
        H_pad = ((H_orig + self.patch_size-1) // self.patch_size) * self.patch_size
        W_pad = ((W_orig + self.patch_size-1) // self.patch_size) * self.patch_size
        x = F.pad(input,(0,W_pad-W_orig,0,H_pad-H_orig),mode='reflect')

        #2.[-1,1]->[0,1]
        x = (x * 0.5 + 0.5-self.imgnet_mean) / self.imgnet_std # TODO 这里没有
        
        # 3. Dino forward
        feature = self.forward_features(x)
        h_grid,w_grid = H_pad // self.patch_size,W_pad // self.patch_size
        if self.rearrange_out:
            feature = feature[:,self.num_prefix_tokens:,:]
            feature = rearrange(feature,"b (h w) c -> b c h w",h=h_grid,w=w_grid)

        # 4.valid_mask:哪些patch完全在原图范围内34x45
        h_valid = H_orig // self.patch_size
        w_valid = W_orig // self.patch_size
        valid_mask = torch.zeros(B, h_grid, w_grid, device=input.device,dtype=torch.bool)
        valid_mask[:,:h_valid,:w_valid] = True
        return feature,valid_mask  # 
    
