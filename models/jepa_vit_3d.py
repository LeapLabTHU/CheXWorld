from torch import nn
from timm.layers.helpers import to_3tuple
import math

class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        img_size=96,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=patch_size,
        )
        self.grid_size = [i//p for i, p in zip(self.img_size, self.patch_size)]
        self.num_patches = math.prod(self.grid_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    