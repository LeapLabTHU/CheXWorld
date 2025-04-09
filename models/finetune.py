from torch import nn
import torch
from timm.models.layers import trunc_normal_
from .attentive_pooler import AttentiveClassifier

class SepFC(nn.Module):
    def __init__(self, in_dim, num_classes, norm=False):
        super().__init__()
        self.fc_list = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim) if norm else nn.Identity(),
                nn.Linear(in_dim, 1)
            )
            for _ in range(num_classes)
        ])
    
    def forward(self, x):
        return torch.cat([fc(x) for fc in self.fc_list], dim=-1)

class LandmarkMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=64, grid_size=196):
        super().__init__()
        self.in_fc = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.out_fc = nn.Linear(hidden_dim * grid_size, num_classes)

    def forward(self, x):
        _b = x.shape[0]
        x = self.act(self.in_fc(x)).view(_b, -1)
        return self.out_fc(x)

class FineTuner(nn.Module):
    def __init__(self, feature_model, feature_dim, num_classes, tune_type='fc', with_cls_token=False):
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.tune_type = tune_type
        self.with_cls_token = with_cls_token
        if tune_type == 'fc':
            self.head = nn.Linear(feature_dim, num_classes)
        elif tune_type == 'sep_fc':
            self.head = SepFC(feature_dim, num_classes)
        elif tune_type == 'sep_fc_norm':
            if hasattr(self.feature_model, 'norm'):
                self.feature_model.norm = nn.Identity()
            self.head = SepFC(feature_dim, num_classes, norm=True)
        elif tune_type == 'drop_fc':
            self.head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(feature_dim, num_classes)
            )
        elif tune_type == 'norm_fc':
            if hasattr(self.feature_model, 'norm'):
                self.feature_model.norm = nn.Identity()
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, num_classes)
            )
            trunc_normal_(self.head[1].weight, std=2e-5)
        elif tune_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Linear(feature_dim // 2, num_classes),
            )
        elif tune_type == 'mlp3':
            self.head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Linear(feature_dim // 2, feature_dim // 4),
                nn.GELU(),
                nn.Linear(feature_dim // 4, num_classes),
            )
        elif tune_type == 'landmark':
            self.head = LandmarkMLP(feature_dim, num_classes)
        elif tune_type == 'attn':
            self.head = AttentiveClassifier(
                embed_dim=feature_dim, num_classes=num_classes
            )
        else:
            raise NotImplementedError
    
    def forward(self, x):
        features = self.feature_model(x)
        if ('fc' in self.tune_type or 'mlp' in self.tune_type) and features.ndim == 3:
            # vit unpooled
            if self.with_cls_token:
                features = features[:, 1:].mean(dim=1)
            else:
                features = features.mean(dim=1)
        return self.head(features)



