"""
NPC Report Generation System - Core Models
===========================================
Contains U-Net and SwinUnet model architectures and tumor analysis classes.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy import ndimage
from pathlib import Path

# Try to import optional dependencies for SwinUnet
try:
    from einops import rearrange
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    SWIN_DEPS_AVAILABLE = True
except ImportError:
    SWIN_DEPS_AVAILABLE = False
    print("Warning: einops or timm not installed. SwinUnet will not be available.")
    print("Install with: pip install einops timm")


# ============================================================
# U-Net Architecture (matching trained checkpoint)
# ============================================================

def conv_block(in_ch, out_ch, dropout_p=0.0):
    """Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU (matching training)"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(),
        nn.Dropout(dropout_p),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU()
    )


class UNet(nn.Module):
    """U-Net matching the trained checkpoint architecture"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_width: int = 16):
        super().__init__()
        
        # Dropout rates matching training
        dropouts = [0.05, 0.1, 0.2, 0.3, 0.5]
        
        # Encoder
        self.encoder1 = conv_block(in_channels, base_width, dropouts[0])        # 1 -> 16
        self.encoder2 = conv_block(base_width, base_width * 2, dropouts[1])     # 16 -> 32
        self.encoder3 = conv_block(base_width * 2, base_width * 4, dropouts[2]) # 32 -> 64
        self.encoder4 = conv_block(base_width * 4, base_width * 8, dropouts[3]) # 64 -> 128
        
        # Bottleneck
        self.bottleneck = conv_block(base_width * 8, base_width * 16, dropouts[4])  # 128 -> 256
        
        # Upsampling (transposed conv)
        self.up4 = nn.ConvTranspose2d(base_width * 16, base_width * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=2, stride=2)
        
        # Decoder (input channels = upsampled + skip connection, no dropout)
        self.decoder4 = conv_block(base_width * 16, base_width * 8, dropout_p=0.0)  # 256 -> 128
        self.decoder3 = conv_block(base_width * 8, base_width * 4, dropout_p=0.0)   # 128 -> 64
        self.decoder2 = conv_block(base_width * 4, base_width * 2, dropout_p=0.0)   # 64 -> 32
        self.decoder1 = conv_block(base_width * 2, base_width, dropout_p=0.0)       # 32 -> 16
        
        # Classifier
        self.classifier = nn.Conv2d(base_width, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))
        
        return self.classifier(d1)
    
    def _match_size(self, x, target):
        if x.size() != target.size():
            x = nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)
        return x


# ============================================================
# SwinUnet Architecture (Swin Transformer for Medical Image Segmentation)
# ============================================================

if SWIN_DEPS_AVAILABLE:
    
    def window_partition(x, window_size):
        """Partition into non-overlapping windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(windows, window_size, H, W):
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class WindowAttention(nn.Module):
        """Window based multi-head self attention"""
        def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = qk_scale or head_dim ** -0.5

            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
            
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
            trunc_normal_(self.relative_position_bias_table, std=.02)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class SwinTransformerBlock(nn.Module):
        """Swin Transformer Block"""
        def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                     act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio
            
            if min(self.input_resolution) <= self.window_size:
                self.shift_size = 0
                self.window_size = min(self.input_resolution)

            self.norm1 = norm_layer(dim)
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

            if self.shift_size > 0:
                H, W = self.input_resolution
                img_mask = torch.zeros((1, H, W, 1))
                h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

        def forward(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)

            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class PatchMerging(nn.Module):
        """Patch Merging Layer (Downsample)"""
        def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

        def forward(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            x = self.reduction(x)
            return x

    class PatchExpand(nn.Module):
        """Patch Expanding Layer (Upsample)"""
        def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
            self.norm = norm_layer(dim // dim_scale)

        def forward(self, x):
            H, W = self.input_resolution
            x = self.expand(x)
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
            x = x.view(B, -1, C // 4)
            x = self.norm(x)
            return x

    class FinalPatchExpand_X4(nn.Module):
        """Final Patch Expanding Layer (4x Upsample)"""
        def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.dim_scale = dim_scale
            self.expand = nn.Linear(dim, 16 * dim, bias=False)
            self.output_dim = dim
            self.norm = norm_layer(self.output_dim)

        def forward(self, x):
            H, W = self.input_resolution
            x = self.expand(x)
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
            x = x.view(B, -1, self.output_dim)
            x = self.norm(x)
            return x

    class BasicLayer(nn.Module):
        """A basic Swin Transformer layer for one stage"""
        def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.use_checkpoint = use_checkpoint

            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

            if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = None

        def forward(self, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class BasicLayer_up(nn.Module):
        """A basic Swin Transformer layer for decoder"""
        def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                     drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.use_checkpoint = use_checkpoint

            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)])

            if upsample is not None:
                self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            else:
                self.upsample = None

        def forward(self, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if self.upsample is not None:
                x = self.upsample(x)
            return x

    class PatchEmbed(nn.Module):
        """Image to Patch Embedding"""
        def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
            super().__init__()
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
            self.img_size = img_size
            self.patch_size = patch_size
            self.patches_resolution = patches_resolution
            self.num_patches = patches_resolution[0] * patches_resolution[1]
            self.in_chans = in_chans
            self.embed_dim = embed_dim
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = norm_layer(embed_dim) if norm_layer is not None else None

        def forward(self, x):
            B, C, H, W = x.shape
            x = self.proj(x).flatten(2).transpose(1, 2)
            if self.norm is not None:
                x = self.norm(x)
            return x

    class SwinTransformerSys(nn.Module):
        """Swin Transformer backbone for U-Net style segmentation"""
        def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=2,
                     embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                     use_checkpoint=False, final_upsample="expand_first", **kwargs):
            super().__init__()

            self.num_classes = num_classes
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.ape = ape
            self.patch_norm = patch_norm
            self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
            self.num_features_up = int(embed_dim * 2)
            self.mlp_ratio = mlp_ratio
            self.final_upsample = final_upsample

            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

            if self.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.absolute_pos_embed, std=.02)

            self.pos_drop = nn.Dropout(p=drop_rate)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            # Encoder
            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint)
                self.layers.append(layer)

            # Decoder
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()
            for i_layer in range(self.num_layers):
                concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                             input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                               patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                             depth=depths[(self.num_layers - 1 - i_layer)],
                                             num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                             window_size=window_size,
                                             mlp_ratio=self.mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             drop=drop_rate, attn_drop=attn_drop_rate,
                                             drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                                             norm_layer=norm_layer,
                                             upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                             use_checkpoint=use_checkpoint)
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)

            self.norm = norm_layer(self.num_features)
            self.norm_up = norm_layer(self.embed_dim)

            if self.final_upsample == "expand_first":
                self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                              dim_scale=4, dim=embed_dim)
                self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward_features(self, x):
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
            x_downsample = []
            for layer in self.layers:
                x_downsample.append(x)
                x = layer(x)
            x = self.norm(x)
            return x, x_downsample

        def forward_up_features(self, x, x_downsample):
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = layer_up(x)
                else:
                    x = torch.cat([x, x_downsample[3 - inx]], -1)
                    x = self.concat_back_dim[inx](x)
                    x = layer_up(x)
            x = self.norm_up(x)
            return x

        def up_x4(self, x):
            H, W = self.patches_resolution
            B, L, C = x.shape
            if self.final_upsample == "expand_first":
                x = self.up(x)
                x = x.view(B, 4 * H, 4 * W, -1)
                x = x.permute(0, 3, 1, 2)
                x = self.output(x)
            return x

        def forward(self, x):
            x, x_downsample = self.forward_features(x)
            x = self.forward_up_features(x, x_downsample)
            x = self.up_x4(x)
            return x

        def load_from(self, pretrained_path):
            """Load pretrained ImageNet weights and map to decoder"""
            if pretrained_path is not None and os.path.exists(pretrained_path):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pretrained_dict = torch.load(pretrained_path, map_location=device, weights_only=False)
                if "model" in pretrained_dict:
                    pretrained_dict = pretrained_dict['model']
                    
                model_dict = self.state_dict()
                full_dict = copy.deepcopy(pretrained_dict)
                
                for k, v in pretrained_dict.items():
                    if "layers." in k:
                        current_layer_num = 3 - int(k[7:8])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k: v})
                        
                for k in list(full_dict.keys()):
                    if k in model_dict:
                        if full_dict[k].shape != model_dict[k].shape:
                            del full_dict[k]
                            
                self.load_state_dict(full_dict, strict=False)

    class SwinUnet(nn.Module):
        """SwinUnet wrapper for medical image segmentation"""
        def __init__(self, img_size=224, num_classes=2, in_chans=1, 
                     embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2],
                     num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                     drop_rate=0., drop_path_rate=0.1):
            super(SwinUnet, self).__init__()
            self.num_classes = num_classes
            self.img_size = img_size
            
            self.swin_unet = SwinTransformerSys(
                img_size=img_size,
                patch_size=4,
                in_chans=3,  # Will repeat 1-channel to 3-channel
                num_classes=num_classes,
                embed_dim=embed_dim,
                depths=depths,
                depths_decoder=depths_decoder,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                final_upsample="expand_first"
            )

        def forward(self, x):
            # Handle 1-channel MRI input -> 3-channel for Swin Transformer
            if x.size()[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            logits = self.swin_unet(x)
            return logits
        
        def load_pretrained(self, path):
            """Load pretrained weights"""
            self.swin_unet.load_from(path)


# ============================================================
# Tumor Features
# ============================================================

@dataclass
class TumorFeatures:
    """Data class for storing tumor features"""
    # Volume metrics
    volume_mm3: float = 0.0
    volume_ml: float = 0.0
    voxel_count: int = 0
    
    # Size metrics
    max_diameter_mm: float = 0.0
    dimensions_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # z, y, x
    
    # Shape metrics
    sphericity: float = 0.0
    elongation: float = 0.0
    surface_area_mm2: float = 0.0
    
    # Location
    centroid_voxel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box: Dict = field(default_factory=dict)
    
    # Status
    tumor_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with native Python types"""
        return {
            'volume_mm3': float(self.volume_mm3),
            'volume_ml': float(self.volume_ml),
            'voxel_count': int(self.voxel_count),
            'max_diameter_mm': float(self.max_diameter_mm),
            'dimensions_mm': tuple(float(d) for d in self.dimensions_mm),
            'sphericity': float(self.sphericity),
            'elongation': float(self.elongation),
            'surface_area_mm2': float(self.surface_area_mm2),
            'centroid_voxel': tuple(float(c) for c in self.centroid_voxel),
            'bounding_box': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) 
                           for k, v in self.bounding_box.items()},
            'tumor_detected': bool(self.tumor_detected)
        }


# ============================================================
# Tumor Analyzer
# ============================================================

class TumorAnalyzer:
    """Analyzes tumor segmentation and extracts features"""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (3.0, 0.5, 0.5)):
        self.voxel_spacing = voxel_spacing  # (z, y, x) in mm
    
    def analyze(self, segmentation: np.ndarray, voxel_spacing: Tuple[float, float, float] = None) -> TumorFeatures:
        """
        Analyze tumor segmentation and extract features.
        
        Args:
            segmentation: Binary mask (D, H, W) where 1 = tumor
            voxel_spacing: Optional (z, y, x) spacing in mm. If None, uses default.
            
        Returns:
            TumorFeatures object
        """
        # Use provided voxel_spacing or default
        if voxel_spacing is not None:
            self.voxel_spacing = tuple(voxel_spacing)
            
        features = TumorFeatures()
        
        # Check if tumor exists
        tumor_mask = segmentation > 0
        features.voxel_count = int(np.sum(tumor_mask))
        
        if features.voxel_count == 0:
            features.tumor_detected = False
            return features
        
        features.tumor_detected = True
        
        # Volume calculation
        voxel_volume = np.prod(self.voxel_spacing)
        features.volume_mm3 = features.voxel_count * voxel_volume
        features.volume_ml = features.volume_mm3 / 1000.0
        
        # Get tumor region
        coords = np.where(tumor_mask)
        
        # Bounding box
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        
        features.bounding_box = {
            'z_min': int(z_min), 'z_max': int(z_max),
            'y_min': int(y_min), 'y_max': int(y_max),
            'x_min': int(x_min), 'x_max': int(x_max)
        }
        
        # Dimensions in mm
        z_dim = (z_max - z_min + 1) * self.voxel_spacing[0]
        y_dim = (y_max - y_min + 1) * self.voxel_spacing[1]
        x_dim = (x_max - x_min + 1) * self.voxel_spacing[2]
        features.dimensions_mm = (float(z_dim), float(y_dim), float(x_dim))
        
        # Max diameter
        features.max_diameter_mm = float(max(features.dimensions_mm))
        
        # Centroid
        centroid_z = float(np.mean(coords[0]))
        centroid_y = float(np.mean(coords[1]))
        centroid_x = float(np.mean(coords[2]))
        features.centroid_voxel = (centroid_z, centroid_y, centroid_x)
        
        # Shape metrics
        features.sphericity = self._calculate_sphericity(tumor_mask)
        features.elongation = self._calculate_elongation(features.dimensions_mm)
        features.surface_area_mm2 = self._calculate_surface_area(tumor_mask)
        
        return features
    
    def _calculate_sphericity(self, mask: np.ndarray) -> float:
        """Calculate sphericity (1 = perfect sphere)"""
        try:
            volume = np.sum(mask) * np.prod(self.voxel_spacing)
            surface_area = self._calculate_surface_area(mask)
            if surface_area > 0:
                sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
                return float(min(1.0, sphericity))
        except:
            pass
        return 0.0
    
    def _calculate_elongation(self, dimensions: Tuple[float, float, float]) -> float:
        """Calculate elongation (ratio of max to min dimension)"""
        dims = sorted(dimensions, reverse=True)
        if dims[-1] > 0:
            return float(dims[0] / dims[-1])
        return 1.0
    
    def _calculate_surface_area(self, mask: np.ndarray) -> float:
        """Estimate surface area using gradient method"""
        try:
            # Use erosion to find surface voxels
            eroded = ndimage.binary_erosion(mask)
            surface = mask & ~eroded
            
            # Count surface voxels and estimate area
            surface_count = np.sum(surface)
            avg_voxel_face = (self.voxel_spacing[0] * self.voxel_spacing[1] + 
                             self.voxel_spacing[1] * self.voxel_spacing[2] + 
                             self.voxel_spacing[0] * self.voxel_spacing[2]) / 3
            
            return float(surface_count * avg_voxel_face)
        except:
            return 0.0


# ============================================================
# Tumor Segmenter
# ============================================================

class TumorSegmenter:
    """Handles tumor segmentation using U-Net model"""
    
    def __init__(self, model: UNet, device: str = "cpu", patch_size: Tuple[int, int] = (256, 256)):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu", 
                            in_channels: int = 1, num_classes: int = 2, 
                            base_width: int = 16,
                            patch_size: Tuple[int, int] = (256, 256)) -> 'TumorSegmenter':
        """Load segmenter from checkpoint file"""
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_width=base_width)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                # Format: {'epoch': ..., 'state_dict': {...}, ...}
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                # Format: {'model_state_dict': {...}, ...}
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the dict is the state_dict itself
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device, patch_size)
    
    @torch.no_grad()
    def segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment a 3D volume slice by slice.
        
        Args:
            volume: 3D numpy array (D, H, W) - assumed to be already z-score normalized
            
        Returns:
            Binary segmentation mask (D, H, W)
        """
        from scipy.ndimage import zoom
        
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        original_h, original_w = volume.shape[1], volume.shape[2]
        target_h, target_w = self.patch_size
        
        # Check if resize is needed
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        for i in range(volume.shape[0]):
            slice_2d = volume[i]
            
            # Resize to patch_size if needed
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            # To tensor
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            # Predict
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            segmentation[i] = pred.astype(np.uint8)
        
        return segmentation
    
    def segment_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Segment a single 2D slice"""
        from scipy.ndimage import zoom
        
        original_h, original_w = slice_2d.shape
        target_h, target_w = self.patch_size
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        with torch.no_grad():
            # Resize to patch_size if needed
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            return pred.astype(np.uint8)


# ============================================================
# SwinUnet Tumor Segmenter
# ============================================================

class SwinUnetSegmenter:
    """Handles tumor segmentation using SwinUnet model"""
    
    def __init__(self, model, device: str = "cpu", patch_size: Tuple[int, int] = (224, 224)):
        """
        Initialize SwinUnet Segmenter.
        
        Args:
            model: SwinUnet model instance
            device: Device to run model on ('cpu' or 'cuda')
            patch_size: Expected input size for SwinUnet (default 224x224)
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path, device: str = "cpu",
                            num_classes: int = 2, 
                            img_size: int = 224,
                            embed_dim: int = 96,
                            depths: List[int] = [2, 2, 2, 2],
                            num_heads: List[int] = [3, 6, 12, 24],
                            window_size: int = 7,
                            patch_size: Tuple[int, int] = (224, 224)) -> 'SwinUnetSegmenter':
        """
        Load SwinUnet segmenter from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to run model on
            num_classes: Number of output classes
            img_size: Input image size (SwinUnet uses 224 by default)
            embed_dim: Embedding dimension
            depths: Depth of each Swin stage
            num_heads: Number of attention heads per stage
            window_size: Window size for attention
            patch_size: Input patch size tuple
        """
        if not SWIN_DEPS_AVAILABLE:
            raise ImportError("SwinUnet requires einops and timm. Install with: pip install einops timm")
        
        # Create SwinUnet model
        model = SwinUnet(
            img_size=img_size,
            num_classes=num_classes,
            in_chans=1,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=[1, 2, 2, 2],
            num_heads=num_heads,
            window_size=window_size
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading directly
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        return cls(model, device, patch_size)
    
    def _normalize_minmax(self, volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Min-Max normalization to [0, 1] as required by SwinUnet"""
        min_val = volume.min()
        max_val = volume.max()
        return (volume - min_val) / (max_val - min_val + epsilon)
    
    @torch.no_grad()
    def segment_volume(self, volume: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Segment a 3D volume slice by slice.
        
        Args:
            volume: 3D numpy array (D, H, W)
            normalize: Whether to apply MinMax normalization (SwinUnet expects [0,1])
            
        Returns:
            Binary segmentation mask (D, H, W)
        """
        from scipy.ndimage import zoom
        
        # Apply MinMax normalization if needed (SwinUnet expects [0, 1])
        if normalize:
            volume = self._normalize_minmax(volume)
        
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        original_h, original_w = volume.shape[1], volume.shape[2]
        target_h, target_w = self.patch_size
        
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        for i in range(volume.shape[0]):
            slice_2d = volume[i]
            
            # Resize to patch_size if needed
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            # To tensor (1-channel, SwinUnet will repeat to 3-channel internally)
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            # Predict
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Resize prediction back to original size
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            segmentation[i] = pred.astype(np.uint8)
        
        return segmentation
    
    def segment_slice(self, slice_2d: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Segment a single 2D slice"""
        from scipy.ndimage import zoom
        
        # Apply MinMax normalization if needed
        if normalize:
            slice_2d = self._normalize_minmax(slice_2d)
        
        original_h, original_w = slice_2d.shape
        target_h, target_w = self.patch_size
        need_resize = (original_h != target_h) or (original_w != target_w)
        
        with torch.no_grad():
            if need_resize:
                slice_resized = zoom(slice_2d, (target_h / original_h, target_w / original_w), order=1)
            else:
                slice_resized = slice_2d
            
            tensor = torch.from_numpy(slice_resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device)
            
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            if need_resize:
                pred = zoom(pred, (original_h / target_h, original_w / target_w), order=0)
            
            return pred.astype(np.uint8)
