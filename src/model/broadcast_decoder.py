import torch
import torch.nn as nn
import torch.nn.functional as F

class BroadcastDecoder(nn.Module):
    def __init__(
            self,
            slot_dim: int,
            dino_dim: int,
            ref_spatial_h: int = 37,
            ref_spatial_w: int = 37,
            hidden_dims: list = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 2048, 2048]

        self.ref_h = ref_spatial_h
        self.ref_w = ref_spatial_w

        # Learnable spatial position embedding (2D grid)
        self.posit_embed = nn.Parameter(
            torch.zeros(1, slot_dim, ref_spatial_h, ref_spatial_w)
        )
        nn.init.trunc_normal_(self.posit_embed, std=1.0)

        # Backbone MLP
        in_dim = slot_dim
        dims = hidden_dims + [dino_dim + 1]

        layers = []
        for i in range(len(dims)):
            layers.append(nn.Linear(in_dim, dims[i]))
            if i < len(dims) - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = dims[i]
        self.backbone = nn.Sequential(*layers)

    def forward(self, slotz, spatial_h, spatial_w, valid_mask=None):
        """
        Args:
            slotz: (B, N, C_slot)
            spatial_h: actual height of feature map
            spatial_w: actual width of feature map
            valid_mask: (B, H*W), True=valid
        Returns:
            recon: (B, H*W, dino_dim)
            alpha: (B, N, H*W)
        """
        B, N, C = slotz.shape
        HW = spatial_h * spatial_w

        # Interpolate position embedding to actual spatial size
        pos = F.interpolate(
            self.posit_embed, size=(spatial_h, spatial_w),
            mode='bilinear', align_corners=False
        )  # (1, slot_dim, h, w)
        pos = pos.reshape(1, C, HW).permute(0, 2, 1)  # (1, HW, C)

        # Broadcast each slot to every spatial position
        mixture = slotz.unsqueeze(2).expand(-1, -1, HW, -1)
        mixture = mixture.reshape(B * N, HW, C)

        # Add position embedding
        mixture = mixture + pos

        # MLP
        output = self.backbone(mixture)
        recon = output[:, :, :-1]
        alpha = output[:, :, -1:]

        recon = recon.view(B, N, HW, -1)
        alpha = alpha.view(B, N, HW, 1)

        alpha = alpha.softmax(dim=1)
        recon = (recon * alpha).sum(dim=1)

        alpha_2d = alpha.squeeze(-1)
        return recon, alpha_2d

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
            print("on cpu MLP,stop")
            exit(1)
