import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import roi_align, roi_pool


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        # assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim)
        self.k_proj = nn.Linear(kv_dim, embed_dim)
        self.v_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query)  # NLC
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Combine heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output


class AttentionPool3d(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, return_attn=False):  # x: BCLHW
        # import pdb;pdb.set_trace()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # BC(LHW) -> (LHW)BC
        x_mean = x.mean(dim=0, keepdim=True)  # (1)BC
        x = torch.cat([x_mean, x], dim=0)  # (LHW+1)BC
        x = x.permute(1, 0, 2).contiguous()  # B(LHW+1)C
        x_mean = x_mean.permute(1, 0, 2).contiguous()  # B(1)C

        if return_attn:
            x, attn = self.cross_attn(query=x_mean, key=x, value=x, return_attn=True)  # B(1)C
            return x.squeeze(dim=-1), attn
        x = self.cross_attn(query=x_mean, key=x, value=x).squeeze(dim=1)  # BC
        batch, channels = x.shape
        x = x.view(batch, channels, 1, 1, 1)

        return x


class TextAttentionPool3d(nn.Module):
    def __init__(self, embed_dim: int, txt_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=txt_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, txt_feat):
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # BC(LHW) -> (LHW)BC
        x_mean = x.mean(dim=0, keepdim=True)  # (1)BC
        x = torch.cat([x_mean, x], dim=0)  # (LHW+1)BC
        x = x.permute(1, 0, 2).contiguous()  # B(LHW+1)C
        x_mean = x_mean.permute(1, 0, 2).contiguous()  # B(1)C

        txt_feat = txt_feat.unsqueeze(dim=1)  # BC -> B(1)C

        x = self.cross_attn(query=txt_feat, key=x, value=x)  # B(1)C
        x = x.squeeze(dim=1)
        batch, channels = x.shape
        x = x.view(batch, channels, 1, 1, 1)
        return x


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
        pre_pool: whether pre-pool the features or not (True for Aesthetic Attributes, False for Technical Attributes)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, pre_pool=False, attn_pool3d=False,
            text_pool3d=False, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_pool = pre_pool
        self.attn_pool3d = attn_pool3d
        self.text_pool3d = text_pool3d
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.attn_pool3d:
            self.attn_pool = AttentionPool3d(embed_dim=self.in_channels, num_heads=12,
                                             output_dim=self.in_channels)  # 768//64=12
        if self.text_pool3d:
            self.text_pool = TextAttentionPool3d(embed_dim=self.in_channels, txt_dim=1024, num_heads=12,
                                                 output_dim=self.in_channels)

        self.fc_hid = nn.Conv3d(2 * self.in_channels, self.hidden_channels,
                                (1, 1, 1)) if self.text_pool3d else nn.Conv3d(self.in_channels, self.hidden_channels,
                                                                              (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

    def forward(self, x, txt=None, inference=False, rois=None):
        # import pdb;pdb.set_trace()
        if self.pre_pool:
            x = self.avg_pool(x)
        if self.attn_pool3d:
            x_vis = self.attn_pool(x)
        if self.text_pool3d and txt is not None:
            x_txt = self.text_pool(x, txt)
            if inference and x_txt.size(0) != x_vis.size(0):
                x_txt = x_txt.expand(x_vis.size(0), -1, -1, -1, -1)
            x = torch.concat([x_vis, x_txt], dim=1)
        if self.attn_pool3d and not self.text_pool3d:
            x = self.dropout(x_vis)
        else:
            x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score


def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output


class VQAHead_cls(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
        pre_pool: whether pre-pool the features or not (True for Aesthetic Attributes, False for Technical Attributes)
    """

    def __init__(
            self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, pre_pool=False, attn_pool3d=False,
            text_pool3d=False, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_pool = pre_pool
        self.attn_pool3d = attn_pool3d
        self.text_pool3d = text_pool3d
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.attn_pool3d:
            self.attn_pool = AttentionPool3d(embed_dim=self.in_channels, num_heads=16,
                                             output_dim=self.in_channels)  # 768//64=12
        if self.text_pool3d:
            self.text_pool = TextAttentionPool3d(embed_dim=self.in_channels, txt_dim=1024, num_heads=16,
                                                 output_dim=self.in_channels)
        # self.fc_hid=nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_hid = nn.Conv3d(2 * self.in_channels, self.hidden_channels,
                                (1, 1, 1)) if self.text_pool3d else nn.Conv3d(self.in_channels, self.hidden_channels,
                                                                              (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.fc_cls1 = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_cls2 = nn.Conv3d(self.hidden_channels, 10, (1, 1, 1))
        self.gelu_cls = nn.GELU()

    def forward(self, x, txt=None, inference=False, rois=None):
        # import pdb;pdb.set_trace()
        if self.pre_pool:
            x = self.avg_pool(x)
        if self.attn_pool3d:
            x_vis = self.attn_pool(x)
        x_cls = self.fc_cls2(self.dropout(self.gelu_cls(self.fc_cls1(x_vis))))
        if self.text_pool3d and txt is not None:
            x_txt = self.text_pool(x, txt)
            if inference and x_txt.size(0) != x_vis.size(0):
                x_txt = x_txt.expand(x_vis.size(0), -1, -1, -1, -1)
            x = torch.concat([x_vis, x_txt], dim=1)
        if self.attn_pool3d and not self.text_pool3d:
            x = self.dropout(x_vis)
        else:
            x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        # print(qlt_score.shape)
        return qlt_score#, x_cls
class VARHead(nn.Module):
    """MLP Regression Head for Video Action Recognition.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(self, in_channels=768, out_channels=400, dropout_ratio=0.5, **kwargs):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        x = self.avg_pool(x)
        out = self.fc(x)
        return out
