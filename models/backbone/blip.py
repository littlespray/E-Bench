'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import warnings

warnings.filterwarnings("ignore")

from blip_models.vit import VisionTransformer, interpolate_pos_embed
from blip_models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import  BertTokenizer
from timm.models.vision_transformer import Attention as TemporalAttention
from timm.layers import Mlp, DropPath, to_2tuple
from timm.layers import PatchEmbed, Mlp, DropPath, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

class MyAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            step:int=1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.step=step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, self.head_dim).permute(3, 1, 0, 4, 2, 5)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        k=torch.cat((k[:self.step,...],k),dim=0)[:int(-1*self.step),...]
        v=torch.cat((v[:self.step,...],v),dim=0)[:int(-1*self.step),...]
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            return attn


from einops import rearrange

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None,type="A"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = TemporalAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop)
        # elif ws == 1:
        #     self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.temporal_attn_1=MyAttention(dim, num_heads,attn_drop=attn_drop,proj_drop=drop,step=1)
        self.temporal_attn_2 = MyAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop,step=2)
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3,stride=1, padding=1)
        self.type=type
        self.gelu=nn.GELU()

    def forward(self, x,B):
        # x: (B*T, h*w, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # spatial
        if self.type=="A":
            temp = self.mlp(self.norm2(x))

            temp=rearrange(temp,'(b t) l c -> b t l c', b=B)

            # step_1=self.drop_path(self.temporal_attn_1(temp))
            # step_2=self.drop_path(self.temporal_attn_2(temp))
            # step=torch.cat((step_2,step_1),dim=1)
            # temp=torch.cat((step,temp),dim=1)
            # temporal
            temp = rearrange(temp, 'b t l c -> (b l) c t', b=B)

            temp = self.temporal_conv(temp)
            temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)

            # output
            x = x + self.drop_path(temp)
        elif self.type=="B":
            spatial = self.mlp(self.norm2(x))
            temp=rearrange(spatial,'(b t) l c->(b l) c t',b=B)
            temp = self.temporal_conv(temp)
            temp = rearrange(temp, '(b l) c t -> (b t) l c', b=B)
            x=x+self.gelu(temp)+self.gelu(spatial)

        #x=rearrange(x,'(b t) l c -> b t l c',b=B).mean(1)
        return rearrange(x,'(b t) l c -> b t l c',b=B).mean(1),rearrange(x,'(b t) l c -> b t l c',b=B)


class My_BLIP_Base(nn.Module):
    def __init__(self,
                 med_config='models/backbone/BLIP_configs/med_config.json',
                 image_size=224,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 drop_path=0.2,
                 in_chans=1024,
                 embed_dim=1024,
                 patch_size=2,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
        self.temporal_block=Block(dim=1024,num_heads=8,drop_path=0.2)
        #self.post_block=PostBlock(t_dim=768,v_dim=1024)
        self.drop_path0 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.softmax=nn.Softmax(dim=1)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        for name, m in self.named_modules():
            if 'temporal_conv' in name:
                nn.init.dirac_(m.weight.data) # initialized to be identity
                nn.init.zeros_(m.bias.data)
            if 'temporal_fc' in name:
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)



    def threeDConv(self,video):#3DConv
        temporal=self.visual_encoder(video)
        return self.temporal_block(temporal.reshape(-1,temporal.shape[-2],temporal.shape[-1]),B=temporal.shape[0])



    def forward(self, video, caption, mode):

        text = self.tokenizer(caption, return_tensors="pt",padding=True).to(video.device)

        assert mode=="multimodal_text"
        image_embeds, frame_embeds = self.threeDConv(video)  # 8,197,1024
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(video.device)

        text.input_ids[:, 0] = self.tokenizer.enc_token_id
        output = self.text_encoder(text.input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        return output.last_hidden_state



def blip_feature_extractor(pretrained='', **kwargs):
    model = My_BLIP_Base(vit="large",**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        #assert (len(msg.missing_keys) == 0)
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing,
                                           ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        print(url_or_filename)
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg

class MyBLIP(nn.Module):
    def __init__(self,type="multimodal"):
        super().__init__()
        self.model = blip_feature_extractor(pretrained="./ckpts/model_large.pth")
        self.type=type



    def forward(self, x, text):
        B, C, T, H, W = x.size()
        return self.model(x,text,self.type).permute(0,2,1).unsqueeze(-1).unsqueeze(-1)

        #return self.model(x, text, "image_attn")

