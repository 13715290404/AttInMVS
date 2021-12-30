from torch import nn
# from timm.models.layers import to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from functools import partial
from einops import rearrange
import torch
import torch.nn.functional as F

from model.base_component import get_sinusoid_encoding_table


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


# Image to Patch Embedding
# MAE:
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding Using Conv2 """

    def __init__(self, image_size=(512, 640), patch_size=(16, 16), input_channels=3, embed_dim=768):
        super().__init__()
        # image_size = to_2tuple(image_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape  # V denotes the number of Input images from different views
        # FIXME look at relaxing size constraints
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size {H}*{W} doesn't match model ({self.image_size[0]} * {self.image_size[1]})"
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activate_layer=nn.GELU):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activate = activate_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
        Multi-head attention, for attention mechanism, 'key' = 'value'
        Inputs:
            query: the 'query' for attention
            value: the 'value' for attention, 'value' has the same dimension as 'query'
        Outputs:
            query: the 'query' after multi-head layer and a linear layer
            value: the 'value' after multi-head attention layer and a linear layer
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, att_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if att_head_dim is not None:
            head_dim = att_head_dim
        all_head_dim = head_dim * self.num_heads
        # 1/head_dim^0.5 in (softmax(Q @ K.T/head_dim^0.5) @ V)
        self.scale = qk_scale or head_dim ** -0.5

        # Before do attention, q,k,v need to do a linear layer
        # There has combined the linear layer corresponding to q,k,v into one linear layer but the output length * 3
        # For key = value, so the input dimension is dim * 2 (query, value)
        self.qkv = nn.Linear(dim * 2, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # The linear layer after multi-head attention and concat
        # Convert the dimension of output(query, value) from all_head_dim to dim (the same as input's dim)
        self.proj = nn.Linear(all_head_dim * 2, dim * 2)

    def forward(self, query, value):
        B, N, C = query.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias])
        x = torch.cat((query, value), dim=-1)  # key = value
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        v = v.reshape(B, N, -1)
        x = torch.cat((x, v), dim=-1)
        x = self.proj(x).reshape(B, N, 2, -1).permute(2, 0, 1, 3)
        v = x[1]
        x = x[0]
        return x, v


class AIMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_scale=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, att_head_dim=None, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, att_head_dim=att_head_dim
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_scale)
        self.mlp = Mlp(input_dim=dim, hidden_dim=mlp_hidden_dim, output_dim=dim, activate_layer=act_layer)

        if init_values is not None and init_values > 0:
            self.gamma1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = None, None

    def forward(self, query, value):
        if self.gamma1 is None:
            x, v = self.attn(self.norm1(query), self.norm1(value))
            # Residual connect
            x = x + query
            v = v + value
            x = x + self.mlp(self.norm2(x))
            v = v + self.mlp(self.norm2(v))
        else:
            x, v = self.gamma1 * self.attn(self.norm1(query), self.norm1(value)) + (query, value)
            # Residual connect
            x = x + query
            v = v + value
            x = x + self.gamma2 * self.mlp(self.norm2(x))
            v = v + self.gamma2 * self.mlp(self.norm2(v))
        return x, v


class MLP(nn.Module):
    def __init__(self, input_dim, norm_layer):
        super().__init__()
        self.input_dim = input_dim
        output_dim = 16 * 16
        self.linear1 = nn.Linear(self.input_dim, output_dim)
        self.mlp_norm1 = norm_layer(output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim // 4)
        self.mlp_norm2 = norm_layer(output_dim // 4)
        self.linear3 = nn.Linear(output_dim // 4, output_dim // 16)

    def forward(self, x):
        # MLP
        x = self.linear1(x)
        x = self.mlp_norm1(x)
        x = self.linear2(x)
        x = self.mlp_norm2(x)
        x = self.linear3(x)
        return x


class AIM(nn.Module):
    def __init__(self,
                 image_size=(512, 640),
                 patch_size=(16, 16),
                 input_channels=3,
                 embed_dim=768,
                 mlp_scale=4.,
                 block_nums=12,
                 num_heads=12,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        # Input embedding
        self.patch_embed = PatchEmbed(image_size, patch_size, input_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embedding
        if use_learnable_pos_emb:
            self.position_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.position_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.position_embed, std=.02)

        # AIM block
        self.blocks = nn.ModuleList([
            AIMBlock(
                dim=embed_dim, num_heads=num_heads, mlp_scale=mlp_scale, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, init_values=init_values
            ) for i in range(block_nums)
        ])

        self.norm = norm_layer(embed_dim)
        self.mlp = MLP(embed_dim, norm_layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, ref, source):
        # Image embedding
        query = self.patch_embed(ref)
        value = self.patch_embed(source)
        # Position embedding
        x = query + self.position_embed
        v = value + self.position_embed
        x_vis, v_vis = x, v

        for blk in self.blocks:
            x_vis, v_vis = blk(x_vis, v_vis)

        x_vis = self.norm(x_vis)

        return x_vis

    def forward(self, images):
        B, N, H, W, C = images.shape
        # assert (B, H, W, C) == ref.shape, \
        #     f"Ref image size: ({ref.shape}) doesn't match source image shape size: ({B}, {H}, {W}, {C})"

        # images_embed = self.patch_embed(images)
        images = images.permute(1, 0, 4, 2, 3)
        x_sum = 0.
        for i in range(1, N):
            x_sum += self.forward_features(images[0], images[i])

        # MLP
        x = self.mlp(x_sum)

        x = rearrange(x, 'b (h w) (p1 p2) -> b (h p1) (w p2)',
                      p1=self.patch_size[0] // 4, p2=self.patch_size[1] // 4,
                      h=self.image_size[0] // self.patch_size[0], w=self.image_size[1] // self.patch_size[1])
        return x


@register_model
def aim_base_patch16_224(pretrained=False, **kwargs):
    model = AIM(
        image_size=(512, 640),
        patch_size=(16, 16),
        input_channels=3,
        embed_dim=768,
        block_nums=8,
        mlp_scale=4.,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    if pretrained:
        checkpoint = torch.load(
            kwargs['init_ckpt'], map_location='cpu'
        )
        model.load_state_dict(checkpoint[model])
    return model
