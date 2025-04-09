from torch import nn
import torch.nn.functional as F
import torch
from .utils import token_to_grid
from timm.models.vision_transformer import VisionTransformer
# from monai.networks.nets import unetr
from .dinov2.models.vision_transformer import DinoVisionTransformer


class BasicBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.SyncBatchNorm(out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.SyncBatchNorm(out_chans),
            nn.GELU(),
        )
        self.skip = nn.Identity() if in_chans == out_chans else nn.Conv2d(in_chans, out_chans, kernel_size=1)

    def forward(self, x):
        return self.conv(x) + self.skip(x)

class DecoderBlock(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv_block = BasicBlock(out_dim*3, out_dim)
    
    def forward(self, x, skip=None):
        out = F.interpolate(x, scale_factor=2, mode="nearest")
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super().__init__()
        self.in_encoder = BasicBlock(3, encoder_channels[0])
        self.decoders = nn.ModuleList([
            DecoderBlock(out_dim=c) for c in encoder_channels[:-1]
        ])
        self.out = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)
        
    def forward(self, features):
        x_in = self.in_encoder(features[0])
        features = features[1:][::-1] # drop the first one and reverse
        
        x = features[0]
        skips = list(features[1:]) + [x_in]
        for skip, decoder in zip(skips, self.decoders[::-1]):
            x = decoder(x, skip)
        
        return self.out(x)


class UNetAdapter_4LayersConv(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=4, out_indices=(11,11,11,11)):
        super().__init__()
        self.feature_model = feature_model
        embed_dim = feature_model.embed_dim
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.encoder_channels = [base_channels * (2 ** i) for i in range(5)]
        self.head = UnetrDecoder(encoder_channels=self.encoder_channels, num_classes=num_classes)
        
        fpn0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 4),
            nn.Conv2d(embed_dim // 4, self.encoder_channels[1], 1),
            nn.SyncBatchNorm(self.encoder_channels[1]),
            nn.GELU(),
            # nn.Upsample(scale_factor=(4,4), mode='bilinear')
        )
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.Conv2d(embed_dim // 2, self.encoder_channels[2], 1),
            nn.SyncBatchNorm(self.encoder_channels[2]),
            nn.GELU(),
            # nn.Upsample(scale_factor=(2,2), mode='bilinear')
        )
        fpn2 = nn.Sequential(
            nn.Conv2d(embed_dim, self.encoder_channels[3], 1),
            nn.SyncBatchNorm(self.encoder_channels[3]),
            nn.GELU()
            # nn.Identity()
        )
        fpn3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, 2, stride=2),
            nn.SyncBatchNorm(embed_dim*2),
            nn.Conv2d(embed_dim*2, self.encoder_channels[4], 1),
            nn.SyncBatchNorm(self.encoder_channels[4]),
            nn.GELU()
            # nn.Upsample(scale_factor=(0.5, 0.5), mode='bilinear')
        )
        self.fpns = nn.ModuleList([fpn0, fpn1, fpn2, fpn3])

    
    def head_parameters(self):
        return list(self.head.named_parameters()) + list(self.fpns.named_parameters())
    
    def forward(self, x):
        if isinstance(self.feature_model, DinoVisionTransformer):
            x_input = F.interpolate(x, (448, 448))
            features = self.feature_model.get_intermediate_layers(x_input, n=12, reshape=False)  #(32, 32)
        elif isinstance(self.feature_model, VisionTransformer):
            features = self.feature_model.get_intermediate_layers(x, n=12, reshape=False)
        else:
            features = self.feature_model(x, layer_results=True)
        features = [token_to_grid(features[idx]) for idx in self.out_indices]
        features_fpn = [fpn(f) for f, fpn in zip(features, self.fpns)]
        features_conv = [x] + features_fpn
        return self.head(features_conv)




class UNetAdapterConv(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=4, out_indices=(11,11,11,11)):
        super().__init__()
        self.feature_model = feature_model
        embed_dim = feature_model.embed_dim
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.encoder_channels = [base_channels * (2 ** i) for i in range(5)]
        self.head = UnetrDecoder(encoder_channels=self.encoder_channels, num_classes=num_classes)
        
        fpn0 = nn.Sequential(
            # nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            # nn.SyncBatchNorm(embed_dim // 2),
            # nn.GELU(),
            # nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            # nn.SyncBatchNorm(embed_dim // 4),
            # nn.GELU(),
            # nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            # nn.SyncBatchNorm(embed_dim // 8),
            # nn.Conv2d(embed_dim // 8, self.encoder_channels[1], 1),
            # nn.SyncBatchNorm(self.encoder_channels[1]),
            # nn.GELU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.SyncBatchNorm(embed_dim // 4),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim // 4,  self.encoder_channels[1], kernel_size=3, padding=1),
            nn.SyncBatchNorm(self.encoder_channels[1]),
            nn.GELU(),
        )
        fpn1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim // 2, self.encoder_channels[2], kernel_size=3, padding=1),
            nn.SyncBatchNorm(self.encoder_channels[2]),
            nn.GELU(),
            # nn.Upsample(scale_factor=(2,2), mode='bilinear')
        )
        fpn2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, self.encoder_channels[3], kernel_size=3, padding=1),
            nn.SyncBatchNorm(self.encoder_channels[3]),
            nn.GELU(),
            # nn.Identity()
        )
        fpn3 = nn.Sequential(
            nn.Conv2d(embed_dim, self.encoder_channels[4], 1),
            nn.SyncBatchNorm(self.encoder_channels[4]),
            nn.GELU()
        )
        self.fpns = nn.ModuleList([fpn0, fpn1, fpn2, fpn3])

    
    def head_parameters(self):
        return list(self.head.named_parameters()) + list(self.fpns.named_parameters())
    
    def forward(self, x):
        if isinstance(self.feature_model, DinoVisionTransformer):
            x_input = F.interpolate(x, (448, 448))
            features = self.feature_model.get_intermediate_layers(x_input, n=12, reshape=False)  #(32, 32)
        elif isinstance(self.feature_model, VisionTransformer):
            features = self.feature_model.get_intermediate_layers(x, n=12, reshape=False)
        else:
            features = self.feature_model(x, layer_results=True)
        features = [token_to_grid(features[idx]) for idx in self.out_indices]
        features_fpn = [fpn(f) for f, fpn in zip(features, self.fpns)]
        features_conv = [x] + features_fpn
        return self.head(features_conv)