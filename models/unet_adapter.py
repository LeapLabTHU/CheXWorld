'''Reference: segmentation_models_pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):


        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


from .utils import token_to_grid
from timm.models.vision_transformer import VisionTransformer
from .dinov2.models.vision_transformer import DinoVisionTransformer

class UNetAdapter(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=5, out_indices=(11,11,11,11,11), no_conv=True):
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.encoder_channels = [base_channels * (2 ** i) for i in range(levels)]
        self.decoder_channels = [base_channels * (2 ** i) for i in range(levels)][::-1]
        self.head = nn.Sequential(
            UnetDecoder(
                encoder_channels=[3,]+self.encoder_channels,
                decoder_channels=self.decoder_channels,
                n_blocks=levels,
                use_batchnorm=True,
                center=False
            ),
            nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        )
        embed_dim = feature_model.embed_dim
        fpn0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, self.encoder_channels[0], kernel_size=2, stride=2),
        )
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, self.encoder_channels[1], kernel_size=2, stride=2),
        )
        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, self.encoder_channels[2], kernel_size=2, stride=2),
        )
        fpn3 = nn.Conv2d(embed_dim, self.encoder_channels[3], kernel_size=1)
        fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(embed_dim, self.encoder_channels[4], kernel_size=1))
        self.fpns = nn.ModuleList([fpn0, fpn1, fpn2, fpn3, fpn4])
        
        self.no_conv = no_conv
        if not no_conv:
            stem = nn.Sequential(*[
                nn.Conv2d(3, self.encoder_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(self.encoder_channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.encoder_channels[0], self.encoder_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(self.encoder_channels[0]),
                nn.ReLU(inplace=True),
            ])
            self.conv_stages = [stem]
            for i in range(levels-1):
                stage = nn.Sequential(*[
                nn.Conv2d(self.encoder_channels[i], self.encoder_channels[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(self.encoder_channels[i+1]),
                nn.ReLU(inplace=True)
                ])
                self.conv_stages.append(stage)
            self.conv_stages = nn.ModuleList(self.conv_stages)

    
    def head_parameters(self):
        if self.no_conv:
            return list(self.head.named_parameters()) + list(self.fpns.named_parameters())
        else:
            return list(self.head.named_parameters()) + list(self.conv_stages.named_parameters()) + list(self.fpns.named_parameters())
    
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
        if self.no_conv:
            features_conv = [x] + features_fpn
        else:
            features_conv = [x]
            for idx, stage in enumerate(self.conv_stages):
                x = stage(x)
                features_conv.append(x + features_fpn[idx])
        return self.head(features_conv)




class UNetAdapter_4Layers(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=4, out_indices=(11,11,11,11)):
        super().__init__()
        self.feature_model = feature_model
        embed_dim = feature_model.embed_dim
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.encoder_channels = [base_channels * (2 ** i) for i in range(levels)]
        # self.encoder_channels = [embed_dim for _ in range(levels)]
        self.decoder_channels = [base_channels * (2 ** i) for i in range(levels)][::-1]
        self.head = nn.Sequential(
            UnetDecoder(
                encoder_channels=[3,]+self.encoder_channels,
                decoder_channels=self.decoder_channels,
                n_blocks=levels,
                use_batchnorm=True,
                center=False
            ),
            nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        )
        
        fpn0 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 4),
            nn.Conv2d(embed_dim // 4, self.encoder_channels[0], 1),
            nn.SyncBatchNorm(self.encoder_channels[0]),
            nn.GELU(),
            # nn.Upsample(scale_factor=(4,4), mode='bilinear')
        )
        fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(embed_dim // 2),
            nn.Conv2d(embed_dim // 2, self.encoder_channels[1], 1),
            nn.SyncBatchNorm(self.encoder_channels[1]),
            nn.GELU(),
            # nn.Upsample(scale_factor=(2,2), mode='bilinear')
        )
        fpn2 = nn.Sequential(
            nn.Conv2d(embed_dim, self.encoder_channels[2], 1),
            nn.SyncBatchNorm(self.encoder_channels[2]),
            nn.GELU()
            # nn.Identity()
        )
        fpn3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, 2, stride=2),
            nn.SyncBatchNorm(embed_dim*2),
            nn.Conv2d(embed_dim*2, self.encoder_channels[3], 1),
            nn.SyncBatchNorm(self.encoder_channels[3]),
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


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x






class UNetAdapterSwin(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=5, out_indices=(11,11,11,11,11)):
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.encoder_channels = [base_channels * (2 ** i) for i in range(levels)]
        self.decoder_channels = [base_channels * (2 ** i) for i in range(levels)][::-1]
        self.head = nn.Sequential(
            UnetDecoder(
                encoder_channels=[3,]+self.encoder_channels,
                decoder_channels=self.decoder_channels,
                n_blocks=levels,
                use_batchnorm=True,
                center=False
            ),
            nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        )
        embed_dim = feature_model.embed_dim
        fpn0 = nn.ConvTranspose2d(embed_dim, self.encoder_channels[0], kernel_size=2, stride=2)
        fpn1 = nn.Conv2d(embed_dim, self.encoder_channels[1], kernel_size=1)
        fpn2 = nn.Conv2d(embed_dim * 2, self.encoder_channels[2], kernel_size=1)
        fpn3 = nn.Conv2d(embed_dim * 4, self.encoder_channels[3], kernel_size=1)
        fpn4 = nn.Conv2d(embed_dim * 8, self.encoder_channels[4], kernel_size=1)
        self.fpns = nn.ModuleList([fpn0, fpn1, fpn2, fpn3, fpn4])
    
    
    def forward_swin_layer(self, x, layer):
        for blk in layer.blocks:
            x = blk(x)
        if layer.downsample is not None:
            x_down = layer.downsample(x)
        else:
            x_down = x
        return x_down, x
    
    def forward_features(self, x):
        x = self.feature_model.patch_embed(x)
        if self.feature_model.ape:
            x = x + self.feature_model.absolute_pos_embed
        x = self.feature_model.pos_drop(x)
        outs = []
        for layer in self.feature_model.layers:
            x, x_pre = self.forward_swin_layer(x, layer)
            outs.append(x_pre)
        return [outs[0]] + outs # we use the first stage twice

    def head_parameters(self):
        return list(self.head.named_parameters()) + list(self.fpns.named_parameters())
    
    def forward(self, x):
        # features = self.forward_features(x)
        features = self.feature_model(x)
        features = [features[0]] + list(features)
        # features = [token_to_grid(feature) for feature in features]
        features_fpn = [fpn(f) for f, fpn in zip(features, self.fpns)]
        features_conv = [x] + features_fpn
        return self.head(features_conv)


class UNetAdapterConvNext(nn.Module):
    def __init__(self, feature_model, num_classes=1, base_channels=32, levels=5, out_indices=(11,11,11,11,11)):
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.out_indices = out_indices
        dims = feature_model.dims
        self.encoder_channels = [dims[0]] + dims
        self.decoder_channels = [base_channels * (2 ** i) for i in range(levels)][::-1]
        self.head = nn.Sequential(
            UnetDecoder(
                encoder_channels=[3,]+self.encoder_channels,
                decoder_channels=self.decoder_channels,
                n_blocks=levels,
                use_batchnorm=True,
                center=False
            ),
            nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=3, padding=1)
        )
        
        # fpn0 = nn.ConvTranspose2d(dims[0], self.encoder_channels[0], kernel_size=2, stride=2)
        # fpn1 = nn.Conv2d(dims[0], self.encoder_channels[1], kernel_size=1)
        # fpn2 = nn.Conv2d(dims[1], self.encoder_channels[2], kernel_size=1)
        # fpn3 = nn.Conv2d(dims[2], self.encoder_channels[3], kernel_size=1)
        # fpn4 = nn.Conv2d(dims[3], self.encoder_channels[4], kernel_size=1)
        # self.fpns = nn.ModuleList([fpn0, fpn1, fpn2, fpn3, fpn4])
    
    
    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.feature_model.downsample_layers[i](x)
            x = self.feature_model.stages[i](x)
            outs.append(x)
        return [F.upsample(outs[0], scale_factor=2)] + outs # we use the first stage twice

    def head_parameters(self):
        return list(self.head.named_parameters()) + list(self.fpns.named_parameters())
    
    def forward(self, x):
        features = self.forward_features(x)
        # features = [token_to_grid(feature) for feature in features]
        # features_fpn = [fpn(f) for f, fpn in zip(features, self.fpns)]
        features_conv = [x] + features
        return self.head(features_conv)



class UNetAdapterResNet(nn.Module):
    def __init__(self, feature_model, num_classes=1):
        super().__init__()
        self.feature_model = feature_model
        self.head = nn.Sequential(
            UnetDecoder(
                encoder_channels=(3, 64, 64, 128, 256, 512),
                decoder_channels=(256, 128, 64, 32, 16),
                n_blocks=5,
                use_batchnorm=True,
                center=False
            ),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        )
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.feature_model.conv1, self.feature_model.bn1, self.feature_model.relu),
            nn.Sequential(self.feature_model.maxpool, self.feature_model.layer1),
            self.feature_model.layer2,
            self.feature_model.layer3,
            self.feature_model.layer4,
        ]

    def head_parameters(self):
        return list(self.head.named_parameters())
    
    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(6):
            x = stages[i](x)
            features.append(x)
        return self.head(features)
