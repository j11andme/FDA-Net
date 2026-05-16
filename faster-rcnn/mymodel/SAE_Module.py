import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

class ChannelDynamicWhitening(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.in_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gn_norm = nn.GroupNorm(1, num_features, affine=False)
        
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 4, num_features, 1),
            nn.Sigmoid() 
        )

        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        alpha = self.gate_net(x)
        x_whitened = alpha * self.in_norm(x) + (1 - alpha) * self.gn_norm(x)
        return x_whitened * self.gamma + self.beta

class FrequencyComplementaryExperts(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=4):
        super().__init__()
        self.num_experts = num_experts
        mid_channels = in_channels // reduction
        
        self.experts = nn.ModuleList([])

        # Expert 1
        self.experts.append(nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels), 
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1)
        ))

        # Expert 2
        self.experts.append(nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, dilation=3, groups=mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1)
        ))

        # Expert 3
        self.experts.append(nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1)
        ))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_experts, 1),
        )

    def forward(self, x):
        logits = self.gate(x) 
        
        weights = torch.sigmoid(logits) # [B, 3, 1, 1]
        
        out_features = 0
        for i, expert in enumerate(self.experts):
            w = weights[:, i:i+1, :, :]
            feat = expert(x)
            out_features = out_features + w * feat
            
        return out_features


class SAE_Module(BaseModule):
    def __init__(self, in_channels, init_cfg=None):
        super().__init__(init_cfg)
        self.whitening = ChannelDynamicWhitening(in_channels)
        
        self.recovery = FrequencyComplementaryExperts(in_channels)
        
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_whitened = self.whitening(x)
        
        x_lost = x - x_whitened

        x_recovered = self.recovery(x_lost)

        return x_whitened + x_recovered * (1 + torch.tanh(self.alpha))