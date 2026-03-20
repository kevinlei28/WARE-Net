import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

try:
    from csms6s import SS2D
except ImportError:
    try:
        from net_vmamba_v1 import SS2D
    except ImportError:
        pass

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: m.bias.data.zero_()
                    
    def forward(self, x):
        d1 = nonlinearity(self.dilate1(x))
        d2 = nonlinearity(self.dilate2(d1))
        d3 = nonlinearity(self.dilate3(d2))
        d4 = nonlinearity(self.dilate4(d3))
        out = x + d1 + d2 + d3 + d4
        return out

class ResidualConvBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.LayerNorm(features, eps=1e-6)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.LayerNorm(features, eps=1e-6)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = out.permute(0, 2, 3, 1)
        out = self.bn1(out)
        out = out.permute(0, 3, 1, 2)
        out = self.act1(out)

        out = self.conv2(out)
        out = out.permute(0, 2, 3, 1)
        out = self.bn2(out)
        out = out.permute(0, 3, 1, 2)
        
        out += shortcut
        out = self.act2(out)
        return out

class VSSBlockFusion(nn.Module):

    def __init__(self, hidden_dim=0, drop_path=0.0, norm_layer=nn.LayerNorm, channel_first=True,
                 ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,
                 ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0, ssm_init="v0",
                 forward_type="v2", mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                 gmlp=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op_x = SS2D(d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
                             act_layer=ssm_act_layer, d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
                             initialize=ssm_init, forward_type=forward_type, channel_first=channel_first)
            self.op_y = SS2D(d_model=hidden_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
                             act_layer=ssm_act_layer, d_conv=ssm_conv, conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
                             initialize=ssm_init, forward_type=forward_type, channel_first=channel_first)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim), mlp_act_layer(), nn.Dropout(mlp_drop_rate),
                nn.Linear(mlp_hidden_dim, hidden_dim), nn.Dropout(mlp_drop_rate)
            )

    def _forward(self, input_x, input_y):
        x, y = input_x, input_y
        target_h, target_w = x.shape[-2], x.shape[-1]

        if self.ssm_branch:
            # Norm
            x_norm = x.permute(0, 2, 3, 1)
            x_norm = self.norm(x_norm).permute(0, 3, 1, 2)
            y_norm = y.permute(0, 2, 3, 1)
            y_norm = self.norm(y_norm).permute(0, 3, 1, 2)
            
            # SS2D Forward
            outs_x = self.op_x(x_norm)
            outs_y = self.op_y(y_norm)
            x_out = outs_x[0] if isinstance(outs_x, (list, tuple)) else outs_x
            y_out = outs_y[0] if isinstance(outs_y, (list, tuple)) else outs_y

            if x_out.dim() == 3:
                B, C, L = x_out.shape
                side = int(math.sqrt(L))
                if x_out.shape[1] == C: x_out = x_out.view(B, C, side, side)
                else: x_out = x_out.transpose(1, 2).view(B, C, side, side)
            
            if y_out.dim() == 3:
                B, C, L = y_out.shape
                side = int(math.sqrt(L))
                if y_out.shape[1] == C: y_out = y_out.view(B, C, side, side)
                else: y_out = y_out.transpose(1, 2).view(B, C, side, side)

            if x_out.shape[-2:] != (target_h, target_w):
                x_out = F.interpolate(x_out, size=(target_h, target_w), mode='bilinear', align_corners=True)
            if y_out.shape[-2:] != (target_h, target_w):
                y_out = F.interpolate(y_out, size=(target_h, target_w), mode='bilinear', align_corners=True)

            x = x + self.drop_path(x_out)
            y = y + self.drop_path(y_out)

        if self.mlp_branch:
            x_norm2 = x.permute(0, 2, 3, 1)
            x_norm2 = self.norm2(x_norm2)
            x = x + self.drop_path(self.mlp(x_norm2).permute(0, 3, 1, 2))
            
            y_norm2 = y.permute(0, 2, 3, 1)
            y_norm2 = self.norm2(y_norm2)
            y = y + self.drop_path(self.mlp(y_norm2).permute(0, 3, 1, 2))
            
        return x, y

    def forward(self, input_x, input_y):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, input_x, input_y)
        else:
            return self._forward(input_x, input_y)

class HybridDecoupledHead(nn.Module):
    def __init__(self, nclass, in_channels_list, features=256, drop_path_rate=0.2):
        super().__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(in_ch, features, 1, 1, 0) for in_ch in in_channels_list
        ])

        self.branch_a_mamba = VSSBlockFusion(
            hidden_dim=features, drop_path=drop_path_rate, channel_first=True,
            ssm_d_state=16, ssm_ratio=2.0
        )
        self.branch_b_dpt = ResidualConvBlock(features)
        self.branch_b_dblock = Dblock(features)

        self.final_fusion = nn.Sequential(
            nn.Conv2d(features, features, 3, 1, 1, bias=False),
            nn.LayerNorm(features, eps=1e-6),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Conv2d(features, nclass, 1, 1, 0)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, out_features, patch_h, patch_w):
        projs = []
        for i, x in enumerate(out_features):
            if x.dim() == 3:
                B, L, C = x.shape
                H = int(math.sqrt(L))
                W = int(math.sqrt(L))
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.projects[i](x)
            projs.append(x)
        
        l1, l2, l3, l4 = projs
        feat_deep_1, feat_deep_2 = self.branch_a_mamba(l4, l3)
        branch_a_out = feat_deep_1 + feat_deep_2

        feat_shallow = l1 + l2
        feat_shallow = self.branch_b_dpt(feat_shallow)

        branch_b_out = self.branch_b_dblock(feat_shallow)
        out = branch_a_out + branch_b_out

        out = out.permute(0, 2, 3, 1) # LN need channel last
        out = self.final_fusion[1](out) # LayerNorm
        out = out.permute(0, 3, 1, 2)
        out = self.final_fusion[2](self.final_fusion[0](out)) # Conv + ReLU

        out = self.output_conv(out)

        out = F.interpolate(out, scale_factor=16, mode='bilinear', align_corners=True)
        
        return out

class dual_decoder(nn.Module):
    def __init__(self, encoder_size='large', nclass=3, features=256, backbone=None):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim 
        in_channels_list = [embed_dim] * 4 

        self.head = HybridDecoupledHead(
            nclass=nclass, 
            in_channels_list=in_channels_list, 
            features=features
        )
        
        self.intermediate_layer_idx = {
            'large': [5, 11, 17, 23]
        }
        self.encoder_size = encoder_size

    def forward(self, x):
        patch_size = 16 
        patch_h, patch_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
        
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
        
        out = self.head(features, patch_h, patch_w)
        return out