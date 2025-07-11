import torch
import torch.nn as nn
import torch.nn.functional as F  
from functools import partial  
from timm.models.layers import DropPath  

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

class HypothesesAttention(nn.Module):  
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):  
        super().__init__()  
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  
        # x shape: [Batch, Num_Hypotheses, Sequence_Length, Embedding_Dim]  
        B, H, N, C = x.shape

        # # Reshape for multi-head attention  
        # q = self.hypothesis_query(x).reshape(B, H, N, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)  
        # k = self.hypothesis_key(x).reshape(B, H, N, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)  
        # v = self.hypothesis_value(x).reshape(B, H, N, self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4)  

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention across hypotheses and sequence  
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)  
        attn = self.attn_drop(attn)  

        # Global hypotheses feature fusion  
        x = (attn @ v).transpose(2, 3).reshape(B, H, N, C)  
        x = self.proj(x)  
        x = self.proj_drop(x)  
        
        return x  

class HypothesesTransformerBlock(nn.Module):  
    def __init__(self,   
                 dim,   
                 num_heads,   
                 mlp_hidden_dim,   
                 qkv_bias=False,   
                 qk_scale=None,   
                 drop=0.,   
                 attn_drop=0.,  
                 drop_path=0.,   
                 act_layer=nn.GELU,   
                 norm_layer=nn.LayerNorm):  
        super().__init__()  
        # Layer Normalization before self-attention  
        self.norm1 = norm_layer(dim)  
        
        # Hypotheses-aware Attention Module  
        self.hypotheses_attn = HypothesesAttention(  
            dim,   
            num_heads=num_heads,   
            qkv_bias=qkv_bias,   
            qk_scale=qk_scale,   
            attn_drop=attn_drop,   
            proj_drop=drop  
        )  
        
        # Drop path for regularization  
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  
        
        # Layer Normalization before MLP (Positioning Encoding)
        self.norm2 = norm_layer(dim)  
        
        # MLP for inter-channel feature communication  
        self.mlp = Mlp(  
            in_features=dim,   
            hidden_features=mlp_hidden_dim,   
            act_layer=act_layer,   
            drop=drop  
        )  

    def forward(self, x):  
        # Global hypotheses feature fusion through attention  
        x = x + self.drop_path(self.hypotheses_attn(self.norm1(x)))  
        
        # Inter-channel feature communication through MLP  
        x = x + self.drop_path(self.mlp(self.norm2(x)))  
        
        return x  

class HypothesesTransformerEncoder(nn.Module):  
    def __init__(self,   
                 depth=3,   
                 embed_dim=512,   
                 mlp_hidden_dim=1024,   
                 num_heads=8,   
                 drop_rate=0.1,   
                 num_hypotheses=5,  
                 sequence_length=27,
                 num_joint=17):
        super().__init__()  
        
        # Learnable Positional Embedding for Hypotheses  
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, num_joint, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth rate  
        drop_path_rate = 0.2  
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        # Hypotheses Transformer Encoder Blocks  
        self.blocks = nn.ModuleList([  
            HypothesesTransformerBlock(  
                dim=embed_dim,   
                num_heads=num_heads,   
                mlp_hidden_dim=mlp_hidden_dim,  
                drop=drop_rate,  
                drop_path=dpr[i]  
            )  
            for i in range(depth)  
        ])  

        # Final Layer Normalization  
        self.norm = nn.LayerNorm(embed_dim)  

    def forward(self, x):  
        # Input shape: [Batch, Num_Hypotheses, Sequence_Length, Embedding_Dim]  
        # Add positional embedding  
        x = x + self.pos_embed  
        x = self.pos_drop(x)  

        # Pass through transformer blocks  
        for blk in self.blocks:  
            x = blk(x)  

        # Final normalization  
        x = self.norm(x)  

        return x