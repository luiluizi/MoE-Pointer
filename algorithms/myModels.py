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

class PointerWithPrior(nn.Module):
    def __init__(self, n_embd, n_options=0):
        super().__init__()
        self.n_options = n_options
        self.n_embd = n_embd
        self.Wb = nn.Linear(n_embd, n_embd, bias=False)  # 不加偏置项
        
        self.prior_shaper = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  ,
            nn.Softplus()
        )
        
        self.initial_gain = 10
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.prior_shaper[-2].bias, self.initial_gain) 

    def forward(self, action_logits, prior_logits): 
        batch_size, n_node = prior_logits.size()
        prior_logits = self.prior_shaper(prior_logits.view(-1,1)).view(batch_size, n_node)
        
        return action_logits + prior_logits

# class Attention(nn.Module):
#     def __init__(self, n_embd, n_head, do_proj_kv=True):
#         super().__init__()
#         assert n_embd % n_head == 0, "Embedding dim must be divisible by head count"
#         self.n_head = n_head
#         self.head_dim = n_embd // n_head
#         self.do_proj_kv = do_proj_kv

#         # Linear projections
#         self.query = nn.Linear(n_embd, n_embd)
#         self.key   = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd) if do_proj_kv else lambda x: x
#         self.proj  = nn.Linear(n_embd, n_embd)

#         # Cache for incremental decoding
#         self.k_cache = None
#         self.v_cache = None

#     def reset_kvcache(self):
#         self.k_cache = None
#         self.v_cache = None

#     def forward(self, query, key, value,
#                 is_causal=False, kmask=None,
#                 use_kvcache=False):
#         B, Lq, D = query.shape
#         Lk = key.shape[1]

#         # Project and reshape to multi-head
#         q = self.query(query).view(B, Lq, self.n_head, self.head_dim).transpose(1, 2) if self.do_proj_kv else query.view(B, Lk, self.n_head, self.head_dim).transpose(1,2)
#         k = self.key(key).view(B, Lk, self.n_head, self.head_dim).transpose(1, 2)
#         v = (self.value(value).view(B, Lk, self.n_head, self.head_dim).transpose(1, 2)) if self.do_proj_kv else value.view(B, Lk, self.n_head, self.head_dim).transpose(1,2)

#         # KV cache (for decoding)
#         if use_kvcache:
#             if self.k_cache is not None:
#                 k = torch.cat([self.k_cache, k], dim=2)
#                 v = torch.cat([self.v_cache, v], dim=2)
#             self.k_cache = k
#             self.v_cache = v

#         # Scaled dot-product
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B, nh, Lq, Lk]

#         # Causal mask
#         if is_causal and Lq > 1:
#             idx = torch.arange(Lq, device=q.device)
#             mask = idx[None, :] > idx[:, None]
#             att = att.masked_fill(mask[None, None, :, :], float('-inf'))

#         # Key padding mask
#         if kmask is not None:
#             att = att.masked_fill(~kmask[:, None, None, :], float('-inf'))

#         # Softmax
#         att = F.softmax(att, dim=-1)

#         # Attention output
#         out = (att @ v).transpose(1, 2).reshape(B, Lq, D)
#         return self.proj(out)
    
class Attention(nn.Module):
    def __init__(self, n_embd, n_head, do_proj_kv=True):
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
        self.k_cache = None
        self.v_cache = None

    def reset_kvcache(self):
        self.k_cache = None
        self.v_cache = None


    def forward(self, query, key, value, is_causal=False, kmask=None, use_kvcache=False):
        B, L, D = query.size() # carefully, when decoding, L == 1, which may cause implicit broadcast.
        q = self.query(query).view(B, query.shape[1], self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L_q, hs)
        k = self.key(key).view(B, key.shape[1], self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L_kv, hs)
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

        att = (q @ k.transpose(-2, -1))* (1.0 / math.sqrt(k.size(-1))) # (B, nh, L_kv, L_q)

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
    """
    Transformer encoder block (self-attn + MLP) without qkfeat.
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.attn(x, x, x, is_causal=False, kmask=mask)
        x = self.ln1(x)
        # Feed-forward
        y = self.mlp(x)
        x = self.ln2(x + y)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = Attention(n_embd, n_head)  # masked self-attn
        self.attn2 = Attention(n_embd, n_head, do_proj_kv=False)  # cross-attn
        self.mlp   = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, dec_mask=None,
                enc_outputs=None, enc_mask=None,
                use_kvcache=False):
        # Masked self-attn
        # x shape [B, 1, n_emb]
        x = x + self.attn1(x, x, x,
                          is_causal=True,
                          kmask=dec_mask,
                          use_kvcache=use_kvcache)
        x = self.ln1(x)
        # Cross-attn to encoder outputs
        x = x + self.attn2(x, enc_outputs, enc_outputs,
                          is_causal=False,
                          kmask=enc_mask)
        x = self.ln2(x)
        # Feed-forward
        y = self.mlp(x)
        x = self.ln3(x + y)
        return x

class Encoder(nn.Module):
    def __init__(self, n_block, n_embd, n_head):
        super().__init__()
        self.blocks = nn.ModuleList(
            EncodeBlock(n_embd, n_head) for _ in range(n_block)
        )

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, n_block, n_embd, n_head):
        super().__init__()
        self.blocks = nn.ModuleList(
            DecodeBlock(n_embd, n_head) for _ in range(n_block)
        )

    def reset_kvcache(self):
        for blk in self.blocks:
            blk.attn1.reset_kvcache()
            blk.attn2.reset_kvcache()

    def forward(self, x, dec_mask=None,
                enc_outputs=None, enc_mask=None,
                use_kvcache=False):
        for blk in self.blocks:
            x = blk(x, dec_mask,
                    enc_outputs, enc_mask,
                    use_kvcache)
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
    
class IdentityDe(nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityDe, self).__init__()
    
    def forward(self, tensor, *args, **kwargs):
        return tensor, tensor
    
    def reset_kvcache(self):
        pass

class NoEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        shape = [*x.shape[:-1], self.embedding_dim]
        return torch.zeros(shape, device=x.device)
    
# ---------- MoE adapter ----------
class SimpleExpert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        # x: [N, D]
        return self.net(x)


class MoEFFN(nn.Module):
    """
    Simple top-1 MoE feed-forward layer for single-device use.
    - d_model: input/output dim
    - d_hidden: intermediate dim inside FFN (e.g. 4*d_model)
    - n_expert: number of experts
    - expert_dropout: dropout inside each expert (optional)
    - aux_loss_coef: coefficient for load-balance loss
    Note: gating can optionally take extra context tensors (stage_emb, vehicle_emb)
    which will be concatenated to the token hidden to compute gate logits.
    """
    def __init__(self, d_model, d_hidden, n_expert=4, expert_dropout=0.0, aux_loss_coef=1e-2):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_expert = n_expert
        self.aux_loss_coef = aux_loss_coef


        self.gate_proj = nn.Linear(d_model, n_expert)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model)
            ) for _ in range(n_expert)
        ])

        self.dropout = nn.Dropout(expert_dropout)

    def forward(self, x, gate_ctx=None):
        """
        x: [B, L, D]
        gate_ctx: optional dict with keys 'stage_emb' and/or 'vehicle_emb' — any can be
                  tensor of shape [B, L, D] or [B, D] or [D]. These (if present) will
                  be concatenated to x when computing gating logits (we'll simply add/concat).
                  Implementation below supports both adding (preferred) or concatenation fallback.
        Returns:
            out: [B, L, D]
            aux: dict with auxiliary metrics (load, prob_mean, aux_loss)
        """
        B, L, D = x.shape
        assert D == self.d_model

        # ---------- compute gate logits ----------
        # We try to fuse contextual info into gating input.
        # Simple approach: if gate_ctx provides stage_emb/vehicle_emb with same dim D,
        # we add them into x (broadcast appropriately) before gating projection.
        g_input = x  # [B, L, D]
        if gate_ctx is not None:
            # If stage/vehicle are provided, unify shapes and add
            for k in ('stage_emb', 'vehicle_emb'):
                if k in gate_ctx and gate_ctx[k] is not None:
                    t = gate_ctx[k]
                    # accept shapes: [B, L, D], [B, D], [D]
                    if t.dim() == 1 and t.shape[0] == D:
                        t = t.view(1, 1, D).expand(B, L, D)
                    elif t.dim() == 2 and t.shape[1] == D:
                        # [B, D] -> expand to [B, L, D]
                        t = t.unsqueeze(1).expand(B, L, D)
                    elif t.dim() == 3 and t.shape == (B, L, D):
                        pass
                    else:
                        # fallback: linear project to D
                        t = t.reshape(B * L, -1)
                        proj = nn.Linear(t.shape[-1], D).to(t.device)
                        t = proj(t).view(B, L, D)
                    g_input = g_input + t

        # compute gate logits and softmax probabilities per token
        gate_logits = self.gate_proj(g_input)  # [B, L, n_expert]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, L, E]

        # ---------- compute top-1 assignment ----------
        flat_probs = gate_probs.view(B * L, self.n_expert)  # [N, E], N = B*L
        top1_idx = torch.argmax(flat_probs, dim=1)  # [N]

        # compute per-expert indices
        outputs = torch.zeros(B * L, D, device=x.device, dtype=x.dtype)
        aux = {}

        # For aux metrics
        prob_mean = flat_probs.mean(dim=0).detach()  # [E]
        # load: fraction of tokens assigned to expert
        assigned_counts = torch.bincount(top1_idx, minlength=self.n_expert).float()  # [E]
        load_frac = assigned_counts / (B * L)

        # dispatch each expert
        N = B * L
        x_flat = x.view(N, D)
        for e in range(self.n_expert):
            mask = (top1_idx == e)
            if mask.sum() == 0:
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(-1)  # indices into N
            expert_in = x_flat[idx]  # [n_e, D]
            expert_out = self.experts[e](expert_in)  # [n_e, D]
            expert_out = self.dropout(expert_out)
            outputs[idx] = expert_out

        out = outputs.view(B, L, D)

        # ---------- auxiliary balance loss (Switch-style approx) ----------
        # Use P_e = mean gate prob for expert e; C_e = fraction of tokens assigned
        P = prob_mean  # [E]
        C = load_frac  # [E]
        # encourage P and C to be aligned and spread: loss = E * sum(P * C)
        aux_loss = self.n_expert * torch.dot(P, C) * self.aux_loss_coef

        aux['aux_loss'] = aux_loss
        aux['gate_prob_mean'] = P
        aux['load_frac'] = C

        # Residual: add input as in typical transformer FFN usage (we leave residual outside)
        return out, aux

# ---------- Modified DecodeBlock using MoEAdapter ----------
class DecodeBlockWithMoE(nn.Module):
    """
    Conservative decode block:
      - masked self-attn (dense)
      - cross-attn (dense)
      - Dense FFN (standard) + small MoE adapter (residual)
    This keeps attention dense and uses MoE only as adapter to augment FFN capacity.
    """
    def __init__(self, n_embd, n_head, moe_experts=8, moe_hidden_ratio=0.75):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.attn1 = Attention(n_embd, n_head)  # masked self-attn
        self.attn2 = Attention(n_embd, n_head, do_proj_kv=False)  # cross-attn

        # MoE adapter: small MLP experts
        moe_hidden = int(4 * n_embd * moe_hidden_ratio)  # scale relative to base FFN hidden dim
        self.moe = MoEFFN(n_embd, moe_hidden, n_experts=moe_experts)

    def forward(self, x, dec_mask=None,
                enc_outputs=None, enc_mask=None,
                use_kvcache=False):
        # Masked self-attn (dense, same as before)
        x = x + self.attn1(x, x, x,
                          is_causal=True,
                          kmask=dec_mask,
                          use_kvcache=use_kvcache)
        x = self.ln1(x)

        # Cross-attn to encoder outputs (dense)
        x = x + self.attn2(x, enc_outputs, enc_outputs,
                          is_causal=False,
                          kmask=enc_mask)
        x = self.ln2(x)

        # Feed-forward (dense path)
        # TODO 加信息
        y, aux = self.moe(x)     # [B, T, D]
        x = self.ln3(x + y)
        # return x and aux_loss for aggregation by the decoder
        return x, aux

# ---------- Modified Decoder that accumulates aux losses ----------
class DecoderWithMoE(nn.Module):
    def __init__(self, n_block, n_embd, n_head, moe_experts=8, moe_hidden_ratio=0.75):
        super().__init__()
        self.blocks = nn.ModuleList(
            DecodeBlock(n_embd, n_head) for _ in range(n_block-2)
        )
        self.blocks_with_moe = nn.ModuleList(
            DecodeBlockWithMoE(n_embd, n_head, moe_experts, moe_hidden_ratio) for _ in range(2)
        )

    def reset_kvcache(self):
        for blk in self.blocks:
            blk.attn1.reset_kvcache()
            blk.attn2.reset_kvcache()

    def forward(self, x, dec_mask=None,
                enc_outputs=None, enc_mask=None,
                use_kvcache=False):
        aux_losses = []
        for blk in self.blocks:
            x = blk(x, dec_mask,
                         enc_outputs, enc_mask,
                         use_kvcache)
        for blk in self.blocks_with_moe:
            x, aux = blk(x, dec_mask,
                         enc_outputs, enc_mask,
                         use_kvcache)
            aux_losses.append(aux)
        # sum auxiliary losses
        total_aux = torch.stack(aux_losses).sum() if len(aux_losses) > 0 else x.new_tensor(0.0)
        print(total_aux)
        return x, total_aux