import torch
import torch.nn as nn
import timm
from einops import rearrange

class DINO2ViT(nn.Module):
    """
    https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8
    """

    def __init__(
        self,
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        in_size=518,
        rearrange=True,
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
        model = timm.create_model(model_name, pretrained=True, img_size=in_size)
        self.in_size = model.patch_embed.img_size[0]
        assert self.in_size == in_size
        self.patch_size = model.patch_embed.patch_size[0]
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
                print(f"[{__class__.__name__}] skip {k}")
                continue
            else:
                print(f"[{__class__.__name__}] copy {k}")
                setattr(self, k, v)
        assert hasattr(self, "num_prefix_tokens")

        __class__._pos_embed = model.__class__._pos_embed
        __class__.forward_features = model.__class__.forward_features

        self.rearrange = rearrange
        self.out_size = in_size // self.patch_size
        assert self.out_size <= 518 // 14

    def forward(self, input):
        """
        input: shape=(b,c,h,w), float
        """
        # with pt.inference_mode(True):  # infer+compile: errors
        feature = self.forward_features(input)
        if self.rearrange:
            feature = feature[:, self.num_prefix_tokens :, :]  # remove class token
            feature = rearrange(feature, "b (h w) c -> b c h w", h=self.out_size)
        return feature  # .clone()
