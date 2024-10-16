import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    H_padded, W_padded = x.shape[2], x.shape[3]

    x = x.view(
        B, C, H_padded // window_size, window_size, W_padded // window_size, window_size
    )

    windows = (
        x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W, B):
    H_padded = math.ceil(H / window_size) * window_size
    W_padded = math.ceil(W / window_size) * window_size
    x = windows.view(
        B,
        H_padded // window_size,
        W_padded // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H_padded, W_padded)
    if H_padded != H or W_padded != W:
        x = x[:, :, :H, :W]
    return x


def ct_dewindow(ct, W, H, window_size):
    bs, N, C = ct.shape
    h_windows = H // window_size
    w_windows = W // window_size
    ct = ct.transpose(1, 2).view(bs, C, N)
    ct = F.adaptive_avg_pool1d(ct, h_windows * w_windows)
    ct = ct.view(bs, C, h_windows, w_windows)
    ct = F.interpolate(ct, size=(H, W), mode="bilinear", align_corners=False)
    return ct


def ct_window(ct, W, H, window_size):
    bs, C, H, W = ct.shape
    ct = ct.view(bs, C, H // window_size, window_size, W // window_size, window_size)
    ct = ct.permute(0, 2, 4, 3, 5, 1).contiguous()
    return ct


class PatchEmbed(nn.Module):
    def _init_(self, in_chans=3, in_dim=64, dim=96):
        super()._init_()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class Downsample(nn.Module):
    def _init_(self, dim, keep_dim=False):
        super()._init_()
        dim_out = dim if keep_dim else 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class ConvBlock(nn.Module):
    def _init_(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        super()._init_()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention(nn.Module):
    def _init_(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        resolution=0,
        seq_length=0,
    ):
        super()._init_()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
            seq_length=seq_length,
        )
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.pos_emb_funct(attn, self.resolution**2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAT(nn.Module):
    def _init_(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1.0,
        window_size=7,
        last=False,
        layer_scale=None,
        ct_size=1,
        do_propagation=False,
    ):
        super()._init_()
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size**2)
        self.norm1 = norm_layer(dim)
        cr_tokens_per_window = ct_size**2 if sr_ratio > 1 else 0
        cr_tokens_total = cr_tokens_per_window * sr_ratio * sr_ratio
        self.cr_window = ct_size
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            seq_length=window_size**2 + cr_tokens_per_window,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.window_size = window_size
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )
        self.gamma4 = (
            nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        )
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                resolution=int(cr_tokens_total**0.5),
                seq_length=cr_tokens_total,
            )
            self.hat_mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.hat_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )
            self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=ct_size**2)
            self.gamma1 = (
                nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            )
            self.gamma2 = (
                nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            )
            self.upsampler = nn.Upsample(size=window_size, mode="nearest")
        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens):
        ct = carrier_tokens
        x = self.pos_embed(x)
        if self.sr_ratio > 1:
            ct = ct_dewindow(
                ct,
                self.cr_window * self.sr_ratio,
                self.cr_window * self.sr_ratio,
                self.cr_window,
            )
            ct = self.hat_pos_embed(ct)
            B, C, H, W = ct.shape
            ct_reshaped = ct.permute(0, 2, 3, 1).reshape(B, -1, C)
            ct_attended = self.hat_attn(self.hat_norm1(ct_reshaped))
            ct_attended = ct_attended.reshape(B, H, W, C).permute(0, 3, 1, 2)
            ct = ct + self.hat_drop_path(self.gamma1 * ct_attended)
            ct_reshaped = ct.permute(0, 2, 3, 1).reshape(B, -1, C)
            ct_mlp = self.hat_mlp(self.hat_norm2(ct_reshaped))
            ct_mlp = ct_mlp.reshape(B, H, W, C).permute(0, 3, 1, 2)
            ct = ct + self.hat_drop_path(self.gamma2 * ct_mlp)
            ct = ct_window(
                ct,
                self.cr_window * self.sr_ratio,
                self.cr_window * self.sr_ratio,
                self.cr_window,
            )
            ct = ct.view(ct.shape[0], -1, ct.shape[-1])
            ct = ct.repeat(x.shape[0] // ct.shape[0], 1, 1)
            x = x.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
            x = torch.cat((ct, x), dim=1)
        x = x + self.drop_path(self.gamma3 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4 * self.mlp(self.norm2(x)))
        if self.sr_ratio > 1:
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size,
                    self.window_size * self.window_size,
                ],
                dim=1,
            )
            total_elements = ctr.numel()
            Bg = ctr.shape[0]
            Hg = ctr.shape[-1]
            Ng = total_elements // (Bg * Hg)
            ct = ctr.reshape(Bg, Ng, Hg)
            if self.last and self.do_propagation:
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    Bg, Hg, int(Ng * 0.5), int(Ng * 0.5)
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        return x, ct


class TokenInitializer(nn.Module):
    def _init_(self, dim, input_resolution, window_size, ct_size=1):
        super()._init_()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.ct_size = ct_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.to_global_feature = nn.Sequential(
            self.pos_embed, nn.AdaptiveAvgPool2d((None, None))
        )

    def forward(self, x):
        x = self.to_global_feature(x)
        B, C, H, W = x.shape
        w_size = self.window_size
        num_windows_h = H // w_size
        num_windows_w = W // w_size
        pad_h = (num_windows_h + 1) * w_size - H if H % w_size != 0 else 0
        pad_w = (num_windows_w + 1) * w_size - W if W % w_size != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2:]
        ct = x.view(B, C, H // w_size, w_size, W // w_size, w_size)
        ct = ct.permute(0, 2, 4, 3, 5, 1).contiguous()
        ct = ct.view(-1, w_size * w_size, C)
        return ct


class PosEmbMLPSwinv2D(nn.Module):
    def _init_(
        self,
        window_size,
        pretrained_window_size,
        num_heads,
        seq_length,
        ct_correct=False,
        no_log=False,
    ):
        super()._init_()
        self.window_size = window_size
        self.num_heads = num_heads
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        if not no_log:
            relative_coords_table *= 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / np.log2(8)
            )
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.seq_length = seq_length
        self.register_buffer("relative_bias", relative_bias)
        self.ct_correct = ct_correct

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor, local_window_size):
        if self.deploy:
            input_tensor += self.relative_bias
            return input_tensor
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            n_global_feature = input_tensor.shape[2] - local_window_size
            if n_global_feature > 0 and self.ct_correct:
                step_for_ct = self.window_size[0] / (n_global_feature**0.5 + 1)
                seq_length = int(n_global_feature**0.5)
                indices = []
                for i in range(seq_length):
                    for j in range(seq_length):
                        ind = (i + 1) * step_for_ct * self.window_size[0] + (
                            j + 1
                        ) * step_for_ct
                        indices.append(int(ind))
                top_part = relative_position_bias[:, indices, :]
                lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
                left_part = relative_position_bias[:, :, indices]
            relative_position_bias = torch.nn.functional.pad(
                relative_position_bias, (n_global_feature, 0, n_global_feature, 0)
            ).contiguous()
            if n_global_feature > 0 and self.ct_correct:
                relative_position_bias = relative_position_bias * 0.0
                relative_position_bias[:, :n_global_feature, :n_global_feature] = (
                    lefttop_part
                )
                relative_position_bias[:, :n_global_feature, n_global_feature:] = (
                    top_part
                )
                relative_position_bias[:, n_global_feature:, :n_global_feature] = (
                    left_part
                )
            self.pos_emb = relative_position_bias.unsqueeze(0)
            self.relative_bias = self.pos_emb
        input_tensor += self.pos_emb
        return input_tensor


class PosEmbMLPSwinv1D(nn.Module):
    def _init_(self, dim, rank=2, seq_length=4, conv=False):
        super()._init_()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(
                nn.Linear(self.rank, 512, bias=True),
                nn.ReLU(),
                nn.Linear(512, dim, bias=False),
            )
        self.grid_exists = False
        self.pos_emb = None
        self.deploy = False
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)
        self.conv = conv

    def switch_to_deploy(self):
        self.deploy = True

    def forward(self, input_tensor):
        if input_tensor.dim() == 4:
            b, c, h, w = input_tensor.shape
            input_tensor = input_tensor.view(b, c, -1).transpose(1, 2)
        seq_length = input_tensor.shape[1]
        if self.deploy:
            return input_tensor + self.relative_bias
        else:
            self.grid_exists = False
        if not self.grid_exists:
            self.grid_exists = True
            seq_length_sqrt = int(seq_length**0.5)
            relative_coords_h = torch.arange(
                0, seq_length_sqrt, device=input_tensor.device, dtype=input_tensor.dtype
            )
            relative_coords_w = torch.arange(
                0, seq_length_sqrt, device=input_tensor.device, dtype=input_tensor.dtype
            )
            relative_coords_table = (
                torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
                .contiguous()
                .unsqueeze(0)
            )
            relative_coords_table -= seq_length_sqrt // 2
            relative_coords_table /= seq_length_sqrt // 2
            self.pos_emb = self.cpb_mlp(
                relative_coords_table.flatten(2).transpose(1, 2)
            )
            self.relative_bias = self.pos_emb
        padding_amount = input_tensor.shape[1] - self.pos_emb.shape[1]
        pos_emb_padded = F.pad(
            self.pos_emb, (0, 0, 0, padding_amount), mode="replicate"
        )
        input_tensor = input_tensor + pos_emb_padded
        if input_tensor.dim() == 3:
            b, hw, c = input_tensor.shape
            h = w = int(hw**0.5)
            input_tensor = input_tensor.transpose(1, 2).view(b, c, h, w)
        return input_tensor


class Mlp(nn.Module):
    def _init_(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super()._init_()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class FasterViTLayer(nn.Module):
    def _init_(
        self,
        dim,
        depth,
        input_resolution,
        num_heads,
        window_size,
        ct_size=1,
        conv=False,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        only_local=False,
        hierarchy=True,
        do_propagation=False,
    ):
        super()._init_()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),
                        layer_scale=layer_scale_conv,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = False
        else:
            sr_ratio = input_resolution // window_size if not only_local else 1
            self.blocks = nn.ModuleList(
                [
                    HAT(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),
                        sr_ratio=sr_ratio,
                        window_size=window_size,
                        last=(i == depth - 1),
                        layer_scale=layer_scale,
                        ct_size=ct_size,
                        do_propagation=do_propagation,
                    )
                    for i in range(depth)
                ]
            )
            self.transformer_block = True
        if (
            len(self.blocks)
            and not only_local
            and input_resolution // window_size > 1
            and hierarchy
            and not self.conv
        ):
            self.global_tokenizer = TokenInitializer(
                dim, input_resolution, window_size, ct_size=ct_size
            )
            self.do_gt = True
        else:
            self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        ct = self.global_tokenizer(x) if self.do_gt else None
        B, C, H, W = x.shape
        if self.transformer_block:
            x = window_partition(x, self.window_size)
        for bn, blk in enumerate(self.blocks):
            x, ct = blk(x, ct)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, H, W, B)
        return x


class ResBlock(nn.Module):
    def _init_(self, in_channels, out_channels, stride=1):
        super()._init_()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.norm1(x1)
        x3 = self.relu(x2)
        out = x3
        x4 = self.conv2(out)
        x5 = self.norm2(x4)
        out = x5
        x_shortcut = self.shortcut(x)
        out += x_shortcut
        return out


class FastEncoder(nn.Module):
    def _init_(
        self,
        feature_dim_s16,
        context_dim_s16,
        feature_dim_s8,
        context_dim_s8,
        dim=64,
        in_dim=64,
        depths=[2, 3, 6],
        window_size=[6, 6, 6],
        ct_size=2,
        mlp_ratio=4,
        num_heads=[2, 4, 8],
        resolution=224,
        drop_path_rate=0.2,
        in_chans=3,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        layer_norm_last=False,
        hat=[False, False, True],
        do_propagation=False,
        **kwargs
    ):
        super()._init_()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None:
            hat = [True] * len(depths)
        self.res_x8 = ResBlock(feature_dim_s8, feature_dim_s8 + context_dim_s8)
        self.res_x16 = ResBlock(feature_dim_s16 * 2, feature_dim_s16 + context_dim_s16)
        self.downsample_1 = Downsample(dim)
        self.downsample_2 = Downsample(dim * 2)
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = FasterViTLayer(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                ct_size=ct_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                input_resolution=int(2 ** (-2 - i) * resolution),
                only_local=not hat[i],
                do_propagation=do_propagation,
            )
            self.levels.append(level)

    def init_pos(self, batch_size, height, width, device, amp):
        ys, xs = torch.meshgrid(
            torch.arange(
                height, dtype=torch.half if amp else torch.float, device=device
            ),
            torch.arange(
                width, dtype=torch.half if amp else torch.float, device=device
            ),
            indexing="ij",
        )
        ys = ys - height / 2
        xs = xs - width / 2
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size, 1, 1, 1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.pos_s16 = self.init_pos(batch_size, height, width, device, amp)

    def forward(self, img):
        x = self.patch_embed(img)
        x = self.levels[0](x)
        x = self.downsample_1(x)
        x = self.levels[1](x)
        x8 = self.res_x8(x)
        x = self.downsample_2(x)
        x = self.levels[2](x)
        x16 = self.res_x16(x)
        return x16, x8


if __name__ == "__main__":
    # W = 224
    # H = 224

    H = 384
    # factor = 1    # For Square Dimensions
    W = H * factor  # 768
    model = FastEncoder(128, 64, 128, 64, resolution=H)
    print("Model initialized")

    # x = torch.rand([2,3,224,224])
    x = torch.rand([2, 3, H, W])
    print(x.shape)
    x16, x8 = model(x)

    print(x16.shape)
    print(x8.shape)
