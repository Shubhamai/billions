from dataclasses import dataclass
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: kv cache, flash attention, gqa, rotary position encoding


@dataclass
class LlamaModelConfig:
    dim: int = 2048
    n_layers: int = 16  # ?

    n_heads: int = 32  # no, of head for query attention
    n_kv_heads: int = (
        8  # no, of head for key and value, it's for grouped query attention
    )

    # When using GQA, the total param are reduced, so we add more params to ffn
    multiple_of: int = 256

    # TODO: not sure what is the purpose of this & multiple_of
    ffn_dim_multiplier: Optional[int] = None

    vocab_size: int = 128256
    norm_eps: float = 1e-05
    theta: float = 500000  # for rotary position encoding

    max_batch_size: int = 32  # for kv cache
    max_seq_len: int = 2048  # for kv cache

    device: str = None


def precompute_theta_pos_freqs(
    head_dim: int, seq_len: int, device: str, theta: float
) -> torch.Tensor:
    # rotary position encoding requires even head_dim
    # TODO: what is head dim ? why not directly change embeddings - (seq_len, dim) instead of dealing with head_dim // 2
    assert head_dim % 2 == 0, (
        "head_dim must be even for rotary position encoding, got %d" % head_dim
    )

    # According to paper (15), theta_i = 10000 ^ (-2(i-1)/dim) for i in [1, 2, ...  dim/2]
    # shape - (head_dim // 2)
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()

    # divide to 1.0 because the power is -2 in (15)
    # shape - (head_dim // 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Contruct the positions ("m" parameter)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position (in seq_length) using outer product
    # shape (seq_len, head_dim // 2)
    freqs = torch.outer(m, theta).float()

    # Compute complex numbers in polar form, c = R * exp(i * m * theta), where R = 1
    # (seq_len, head_dim // 2) -> (seq_len, head_dim // 2)
    # TODO: what is polar, understand polar form, why convert to polar and how it's converted
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


# Very clear is figure 1 of the paper
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch_size, seq_len, head, head_dim) -> (batch_size, seq_len, head, head_dim // 2)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )  # TODO: understand this transformation
    # (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # (batch_size, seq_len, head, head_dim // 2) * (1, seq_len, 1, head_dim // 2) -> (batch_size, seq_len, head, head_dim // 2)
    x_rotated = x_complex * freqs_complex

    # (batch_size, seq_len, head, head_dim // 2) -> (batch_size, seq_len, head, head_dim // 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # (batch_size, seq_len, head, head_dim // 2, 2) -> (batch_size, seq_len, head, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-05):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (batch_size, seq_len, dim) * (batch_size, seq_len, 1) -> (batch_size, seq_len, dim)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (dim) * (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            # (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class FeedForward(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()

        # why 4 * config.dim
        hidden_dim = 4 * config.dim

        # why this calculation ?
        hidden_dim = int(2 * hidden_dim / 3)

        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        # why such configuration ?
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class SelfAttention(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()

        self.config = config

        # No. of heads for key and value (different from query due to grouped query attention)
        self.n_kv_heads = config.n_kv_heads

        # No. of heads for query
        self.n_query_heads = config.n_heads

        # No. of heads for key and value
        self.n_rep = self.n_query_heads // self.n_kv_heads

        # Embeddings are split into individual query heads, each head can see all the sequence but part of the dim
        self.head_dim = config.dim // config.n_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # self.cache_k = torch.zeros(
        #     (config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
        # ).to(config.device)
        # self.cache_v = torch.zeros(
        #     (config.max_batch_size, config.max_seq_len, self.n_kv_heads, self.head_dim)
        # ).to(config.device)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
    ):
        # (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_query_heads * head_dim)
        xq = self.wq(x)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (batch_size, seq_len, n_query_heads * head_dim) -> (batch_size, seq_len, n_query_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_query_heads, self.head_dim)

        # (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings to the query and key, does not affect the shape
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        # TODO: will need to disable the cache when training
        # why :batch_size, need to fully understand this part
        # Likely to replace the entire cache with the new values for the current batch
        # self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        # self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve the key and value from the cache
        # (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads * head_dim)
        # keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # values = self.cache_v[:batch_size, : start_pos + seq_len]

        # repeat the head and query for the key and value
        # TODO: it's computation inefficient, need to optimize code to actually take advantage of grouped query attention
        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        # (batch_size, seq_len, n_query_heads, head_dim) -> (batch_size, n_query_heads, seq_len, head_dim)
        # may have some doubt on this, need to understand this
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # why transpost of keys, need to understand this, need to check if below shape transformation is correct
        # # (batch_size, n_query_heads, seq_len, head_dim) @ (batch_size, n_kv_heads, head_dim, seq_len_kv) -> (batch_size, n_query_heads, seq_len, seq_len_kv)
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # if mask is not None:
        #     scores = scores + mask

        # scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # # (batch_size, n_query_heads, seq_len, seq_len_kv) @ (batch_size, n_kv_heads, seq_len_kv, head_dim) -> (batch_size, n_query_heads, seq_len, head_dim)
        # output = torch.matmul(scores, values)

        # supports flash attention 2, not flash attention 3
        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            is_causal=True,  # attn_mask=mask
        )

        # why contiguous, need to understand this
        # (batch_size, n_query_heads, seq_len, head_dim) -> (batch_size, seq_len, n_query_heads, head_dim)
        # (batch_size, seq_len, n_query_heads, head_dim) -> (batch_size, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)


class Block(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()

        self.config = config

        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        self.attention = SelfAttention(config)
        self.ffn = FeedForward(config)

        self.pre_attention_norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.pre_ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_complex: torch.Tensor,
    ):
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        h = x + self.attention.forward(self.pre_attention_norm(x), freqs_complex)

        out = h + self.ffn.forward(self.pre_ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, config: LlamaModelConfig):
        super().__init__()  # TODO: why need it, what if I don't use it

        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(Block(config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # for rotary position encoding
        # TODO: why *2, and what if this for, explained here https://youtu.be/oM4VmoabDAI?si=seaccUdKC41X43ai&t=1968, but still not sure
        self.freqs_complex = precompute_theta_pos_freqs(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            device=config.device,
            theta=config.theta,
        )

    # TODO: what is start_pos
    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len = tokens.shape

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        tokens = self.token_embeddings(tokens)

        # TODO: not sure what this means, make it clear
        # Retrive the pairs (m, theta) corresponding to the current position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[:seq_len]

        for layer in self.layers:
            tokens = layer(tokens, freqs_complex)

        # RMS norm
        tokens = self.norm(tokens)

        if targets is not None:
            logits = self.output(tokens).float()  # why float ?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # logits = self.output(tokens[: [-1], :])
            logits = self.output(tokens).float()  # why float ?
            loss = None

        return logits, loss

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        # L, H, Q, T = cfg.n_layers, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.max_seq_len
                else idx[:, -self.config.max_seq_len :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
