import math

import torch
from torch import nn
from torch.nn import functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

# Only Linear with ReLU use init_
def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

def assign_symmetric(matrix, slice_n, slice_m, sub_matrix):
    if len(sub_matrix.shape) == 0:
        sub_matrix_T = sub_matrix
    else:
        sub_matrix_T = sub_matrix.transpose(-2, -3)
        
    matrix[:, slice_n, slice_m] = sub_matrix
    matrix[:, slice_m, slice_n] = sub_matrix_T


class BatchNorm(nn.Module):
    """
    Fake BatchNorm.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return x

class Attention(nn.Module):

    def __init__(self, n_embd, n_head, do_proj_kv=True, qkfeat_dim=None):
        super(Attention, self).__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.do_proj_kv = do_proj_kv
        self.key = nn.Linear(n_embd, n_embd)
        if do_proj_kv:
            self.query = nn.Linear(n_embd, n_embd)
            self.value = nn.Linear(n_embd, n_embd)
        else:
            self.query = lambda x: x
            self.value = lambda x: x
        self.proj = nn.Linear(n_embd, n_embd)

        if qkfeat_dim is not None:
            self.qkfeat_dim = qkfeat_dim
            self.qkfeat_proj = nn.Linear(qkfeat_dim, n_head)
        else:
            self.qkfeat_dim = n_head
            self.qkfeat_proj = None

        self.k_cache = None
        self.v_cache = None

    def reset_kvcache(self):
        self.k_cache = None
        self.v_cache = None


    def forward(self, query, key, value, qkfeat=None, is_causal=False, kmask=None, use_kvcache=False):
        B, L, D = query.size() # carefully, when decoding, L == 1, which may cause implicit broadcast.

        q = self.query(query).view(B, query.shape[1], self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L_kv, hs)
        k = self.key(key).view(B, key.shape[1], self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L_q, hs)
        v = self.value(value).view(B, value.shape[1], self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L_kv, hs)

        if use_kvcache:
            if self.k_cache is not None:
                k = torch.cat((k, self.k_cache), dim=2)
                v = torch.cat((v, self.v_cache), dim=2)
            self.k_cache = k
            self.v_cache = v
        else:
            if self.k_cache is not None:
                print("warning, k_cache is not None but use_kvcache=False")

        att = (q @ k.transpose(-2, -1)) # (B, nh, L_kv, L_q)
        # edge feature implementation abolited.
        # assert qkfeat.shape == torch.Size([B, L, k.shape[-2], q.shape[-1]])
        # headed_qkfeat = qkfeat.view(B, L, k.shape[-2], self.n_head, D // self.n_head).permute(0, 3, 1, 2, 4) # (B, nh, L_q, L_kv, hs)
        # att = (q.unsqueeze(-2) * (headed_qkfeat + k.unsqueeze(2))).sum(-1)
        if qkfeat is not None:
            # TODO: 这里可能导致训练不稳定吗？为什么不加 use_heur_req 效果如此差。
            assert qkfeat.shape == torch.Size((B, L, k.shape[-2], self.qkfeat_dim))
            # if use_kvcache is False:
            #     print(att.abs().mean(), qkfeat.abs().mean())
            if self.qkfeat_proj is not None:
                qkfeat = self.qkfeat_proj(qkfeat)
            # _norm_debug = att.abs().mean() / qkfeat.abs().mean()
            # print(_norm_debug)
            att = att + qkfeat.permute(0, 3, 1, 2) * 1. # this hyper-parameters is set by make both scale similar, which is 10.0
        
        att = att * (1.0 / math.sqrt(k.size(-1)))

        if is_causal and L > 1:
            mask_out = torch.arange(att.shape[-1], device=q.device) > (torch.arange(att.shape[-2], device=q.device) + att.shape[-1] - att.shape[-2])[:, None]
            att[mask_out.expand_as(att)] = float('-inf')
        
        # 掩码掉不可见的attention score
        if kmask is not None:
            att.masked_fill_(~kmask[:, None, None, :], float('-inf'))
        
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L_q, L_kv) x (B, nh, L_kv, hs) -> (B, nh, L_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)

        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, qkfeat_dim=None):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, qkfeat_dim=qkfeat_dim)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            nn.Linear(1 * n_embd, n_embd)
        )

    def forward(self, x, mask=None, rel_mat=None):
        x = self.ln1(x + self.attn(x, x, x, rel_mat, is_causal=False, kmask=mask))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, qkfeat_dim=None):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = Attention(n_embd, n_head, qkfeat_dim=qkfeat_dim)
        self.attn2 = Attention(n_embd, n_head, do_proj_kv=False, qkfeat_dim=qkfeat_dim)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 4 * n_embd), activate=True),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, dec_mask=None, rel_mat=None, rep_enc=None, enc_mask=None, enc_rel_mat=None, use_kvcache=False):
        x = self.ln1(x + self.attn1(x, x, x, rel_mat, is_causal=True, kmask=dec_mask, use_kvcache=use_kvcache))
        x = self.ln2(x + self.attn2(x, rep_enc, rep_enc, enc_rel_mat, kmask=enc_mask))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, n_block, n_embd, n_head, qkfeat_dim=None):
        super(Encoder, self).__init__()

        self.n_embd = n_embd
        self.blocks = nn.ModuleList(EncodeBlock(n_embd, n_head, qkfeat_dim=qkfeat_dim) for _ in range(n_block))
    
    def forward(self, x, mask=None, rel_mat=None):
        for block in self.blocks:
            x = block(x, mask, rel_mat)
        return x

class Decoder(nn.Module):

    def __init__(self, n_block, n_embd, n_head, qkfeat_dim=None):
        super(Decoder, self).__init__()

        self.n_embd = n_embd
        self.blocks = nn.ModuleList(DecodeBlock(n_embd, n_head, qkfeat_dim=qkfeat_dim) for _ in range(n_block))
    

    def reset_kvcache(self):
        for block in self.blocks:
            block.attn1.reset_kvcache()
            block.attn2.reset_kvcache()

    def forward(self, x, dec_mask=None, rel_mat=None, rep_enc=None, enc_mask=None, enc_rel_mat=None, use_kvcache=False):
        for block in self.blocks:
            x = block(x, dec_mask, rel_mat, rep_enc, enc_mask, enc_rel_mat, use_kvcache=use_kvcache)
        return x

class MLP(nn.Module):
    def __init__(self, *dims, pre_act=False, scale=1., enable_scale=True):
        super().__init__()
        self.seq = nn.Sequential()
        if pre_act:
            self.seq.append(nn.LeakyReLU())
        self.scale = scale
        self.enable_scale = enable_scale
        for i in range(len(dims) - 1):
            self.seq.append(nn.Linear(dims[i], dims[i+1]))
            if i + 2 != len(dims):
                self.seq.append(nn.LeakyReLU())
    
    def forward(self, x):
        if self.scale != 1 and self.enable_scale:
            x = x * self.scale
        return self.seq(x)

class IterMixin:
    def __init__(self):
        self.register_buffer("iter", torch.tensor(0))
    
    def print_iter(self):
        print(self.iter.item())

class MulConstant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        grad_a = grad_output  # 直接传递梯度而不是乘以 b
        return grad_a, None

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    
    def forward(self, tensor, *args, **kwargs):
        return tensor
    
    def reset_kvcache(self):
        pass

class NoEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        shape = [*x.shape[:-1], self.embedding_dim]
        return torch.zeros(shape, device=x.device)