import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def S_Correction(x_pred, x_input):
    x_fft = torch.fft.rfft(x_pred, dim=-1, norm='ortho')
    x_pre_fft = torch.fft.rfft(x_input, dim=-1, norm='ortho')
    
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    
    x_ifft = torch.fft.irfft(x_fft, dim=-1)
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=-1)
    
    x_ifft = torch.clamp(x_ifft, min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft, min=0)
    
    numerator = torch.sum(x_ifft * x_pre_ifft, dim=-1, keepdim=True)
    denominator = torch.sum(x_pre_ifft * x_pre_ifft, dim=-1, keepdim=True) + 1e-5
    
    alpha = numerator / denominator
    alpha = torch.clamp(alpha, min=0, max=1e5) 
    
    return torch.sqrt(alpha + 1e-8)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = TokenEmbedding(c_in=patch_len, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        x = x.reshape(B * C, -1, self.patch_len)
        ve = self.value_embedding(x)
        pe = self.position_embedding(ve)
        x = ve + pe
        x = self.dropout(x)
        x = x.reshape(B, C, -1, x.shape[-1])
        return x

class LinearTemporalModeling(nn.Module):
    def __init__(self, patch_num, d_model):
        super().__init__()
        self.linear_time = nn.Linear(patch_num, patch_num)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x_in = x
        x = x.permute(0, 1, 3, 2)
        x = self.linear_time(x)
        x = self.act(x)
        x = x.permute(0, 1, 3, 2)
        return self.norm(x + x_in)

class ChannelAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.reshape(B, N, C, D).permute(0, 2, 1, 3)
        return self.norm(x + self.dropout(attn_out))

class FullPatchAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, N, D = x.shape
        x_reshaped = x.reshape(B * C, N, D)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.reshape(B, C, N, D)
        return self.norm(x + self.dropout(attn_out))

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        
        self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
        
        self.revin = RevIN(configs.c_out)
        self.patch_embed = PatchEmbedding(configs.d_model, configs.patch_len, configs.stride, configs.dropout)
        
        self.linear_time_model = LinearTemporalModeling(self.patch_num, configs.d_model)
        self.channel_attn_model = ChannelAttention(configs.d_model, n_heads=4, dropout=configs.dropout)
        
        self.fusion_norm = nn.LayerNorm(configs.d_model)
        
        self.full_patch_attn = FullPatchAttention(configs.d_model, n_heads=4, dropout=configs.dropout)
        
        self.head = nn.Flatten(start_dim=-2)
        self.projection = nn.Linear(self.patch_num * configs.d_model, configs.pred_len)

    def forward(self, x_input, x_mark_input=None):
        x_norm = self.revin(x_input, 'norm')
        
        x_patch = self.patch_embed(x_norm)
        x_patch_raw = x_patch.clone()
        
        x_time = self.linear_time_model(x_patch)
        x_chan = self.channel_attn_model(x_patch)
        
        x_fused = self.fusion_norm(x_time + x_chan)
        
        x_out = self.full_patch_attn(x_fused)
        
        correction_factor = S_Correction(
            x_out.permute(0, 1, 3, 2),
            x_patch_raw.permute(0, 1, 3, 2)
        )
        x_out = x_out * correction_factor.permute(0, 1, 3, 2)
        x_out = self.head(x_out)
        x_out = self.projection(x_out)
        
        x_out = x_out.permute(0, 2, 1)
        x_out = self.revin(x_out, 'denorm')
        
        return x_out