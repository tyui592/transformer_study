# -*- coding: utf-8 -*-
"""Transformer Model Code."""

import torch
import torch.nn as nn
import logging
from typing import Optional


def calc_pos_encoding(max_len, d_model):
    """Make a Positional encoding."""
    encoding = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
    _2i = torch.arange(0, d_model, step=2).float()
    encoding[:, 0::2] = torch.sin(pos / (10_000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10_000 ** (_2i / d_model)))
    return encoding


class MHA(nn.Module):
    """Multi-Head Attention Module."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 p_drop: float = 0.1):
        """Init."""
        super().__init__()
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)

        self.scale = torch.sqrt(torch.FloatTensor([d_model / n_heads]))

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Forward.

        mask: boolean mask to ignore future words in decoder during training.
        """
        #  x: (batch) x (# of tokens) x (d_model)
        batch_size, _, d_model = q.shape
        d_head = d_model // self.n_heads

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # Current Tensor Dim: (batch) x (# of tokens) x (d_model)
        # (1) split (d_model) to (n_heads) x (head_dim)
        #      -> (batch) x (# of tokens) x (n_heads) x (head_dim)
        # (2) transpose dimensions for multiplication
        #      -> (batch) x (n_heads) x (# of tokens) x (head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, d_head).transpose(1, 2)

        # calculate attention weights,
        #   Dim: (batch) x (n_heads) x (# of tokens) x (# of tokens)
        attention_score = torch.matmul(Q, K.transpose(2, 3))
        attention_score /= self.scale.to(attention_score.device)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)
        attention_weight = torch.softmax(attention_score, dim=-1)

        # y: (batch) x (n_heads) x (# of tokens) x (head_dim)
        y = torch.matmul(self.dropout(attention_weight), V)

        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, -1, d_model)
        y = self.w_o(y)
        return y, attention_weight


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 p_drop: float = 0.1):
        """init."""
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        """Forward."""
        x = self.dropout(torch.relu(self.w1(x)))
        x = self.w2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder Layer."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1):
        """Init.

        d_model: dimension of model.
        d_ff: dimension of feedforward layers.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)  # for mha
        self.layer_norm2 = nn.LayerNorm(d_model)  # for feed forward

        self.mha = MHA(d_model, n_heads, attention_drop)
        self.ff = FeedForward(d_model, d_ff, residual_drop)
        self.dropout = nn.Dropout(residual_drop)  # residual dropout

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward with q, k, v and mask."""
        _x, _ = self.mha(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(_x))

        _x = self.ff(x)
        x = self.layer_norm2(x + self.dropout(_x))

        return x


class Encoder(nn.Module):
    """Encoder."""

    def __init__(self,
                 n_layers: int = 6,
                 d_inp: int = 100,
                 d_model: int = 512,
                 n_heads: int = 6,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1,
                 embedding_drop: float = 0.1,
                 pos_encoding: str = 'embedding',
                 max_len: int = 100):
        """Init.

        d_inp: Dimension of input. It is depend on the size of source vocab.
        max_len: Maximum number of token.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(d_inp, d_model)
        if pos_encoding == 'embedding':
            self.pos_embedding = nn.Embedding(max_len, d_model)

        elif pos_encoding == 'sinusoid':
            self.pos_embedding = calc_pos_encoding(max_len, d_model)

        else:
            raise NotImplementedError(f"Not expected embedding: {pos_encoding}")

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         n_heads=n_heads,
                         d_ff=d_ff,
                         attention_drop=attention_drop,
                         residual_drop=residual_drop) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(embedding_drop)  # embedding dropout

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Forward."""
        batch_size, num_token = x.shape[:2]

        # make a input tensor for positional embedding
        # -> torch.arange(0, num_token) [num_token]
        # -> unsqueeze: [1, num_token]
        # -> repeat: [batch_size, num_token]
        if isinstance(self.pos_embedding, nn.Embedding):
            pos = torch.arange(0, num_token).unsqueeze(0)
            pos = pos.to(x.device)
            pos_embedding = self.pos_embedding(pos)
        else:
            pos_embedding = self.pos_embedding[:num_token].unsqueeze(0)
            pos_embedding = pos_embedding.to(x.device)
        x = self.dropout(self.token_embedding(x) + pos_embedding)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    """Decoder Layer."""

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1):
        """Init."""
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)  # for first mha
        self.layer_norm2 = nn.LayerNorm(d_model)  # for second mha
        self.layer_norm3 = nn.LayerNorm(d_model)  # for feed forward

        self.mha = MHA(d_model=d_model,
                       n_heads=n_heads,
                       p_drop=attention_drop)

        self.encoder_mha = MHA(d_model=d_model,
                               n_heads=n_heads,
                               p_drop=attention_drop)  # encoder-decoder attention
        self.ff = FeedForward(d_model=d_model,
                              d_ff=d_ff,
                              p_drop=residual_drop)
        self.dropout = nn.Dropout(residual_drop)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                dec_mask: Optional[torch.Tensor] = None,
                enc_mask: Optional[torch.Tensor] = None):
        """Forward.

        memory: memory of encoder.
        dec_mask: mask to ignore future words.
        dnc_mask: mask for encoder-decoder attnetion module.
        """
        _x, _ = self.mha(x, x, x, dec_mask)
        x = self.layer_norm1(x + self.dropout(_x))

        _x, attention_weight = self.encoder_mha(x, memory, memory, enc_mask)
        x = self.layer_norm2(x + self.dropout(_x))

        _x = self.ff(x)
        x = self.layer_norm3(x + self.dropout(_x))

        return x, attention_weight


class Decoder(nn.Module):
    """Decoder."""

    def __init__(self,
                 d_out: int = 100,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1,
                 embedding_drop: float = 0.1,
                 pos_encoding: str = 'embedding',
                 max_len: int = 100):
        """Init.

        d_out: It denpends on the size of target vocab.
        """
        super().__init__()

        self.token_embedding = nn.Embedding(d_out, d_model)
        if pos_encoding == 'embedding':
            self.pos_embedding = nn.Embedding(max_len, d_model)

        elif pos_encoding == 'sinusoid':
            self.pos_embedding = calc_pos_encoding(max_len, d_model)

        else:
            raise NotImplementedError(f"Not expected embedding: {pos_encoding}")

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model,
                         n_heads=n_heads,
                         d_ff=d_ff,
                         attention_drop=attention_drop,
                         residual_drop=residual_drop) for _ in range(n_layers)
        ])
        self.w_out = nn.Linear(d_model, d_out)
        self.dropout = nn.Dropout(embedding_drop)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                dec_mask: Optional[torch.Tensor] = None,
                enc_mask: Optional[torch.Tensor] = None):
        """Forward.

        x: decoder input.
        memory: context vector of encoder.
        dec_mask: mask for training (next token and pad token)
        enc_mask: mask for pad token
        """
        batch_size, num_token = x.shape[:2]

        if isinstance(self.pos_embedding, nn.Embedding):
            pos = torch.arange(0, num_token).unsqueeze(0)
            pos = pos.to(x.device)
            pos_embedding = self.pos_embedding(pos)
        else:
            pos_embedding = self.pos_embedding[:num_token].unsqueeze(0)
            pos_embedding = pos_embedding.to(x.device)
        x = self.dropout(self.token_embedding(x) + pos_embedding)

        for layer in self.layers:
            x, attention = layer(x, memory, dec_mask, enc_mask)

        out = self.w_out(x)
        return out, attention


class Transformer(nn.Module):
    """Transformer."""

    def __init__(self,
                 d_inp: int = 100,
                 d_out: int = 100,
                 d_model: int = 512,
                 d_ff: int = 2048,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 attention_drop: float = 0.0,
                 residual_drop: float = 0.1,
                 embedding_drop: float = 0.1,
                 max_len: int = 100,
                 pos_encoding: str = 'embedding',
                 pad_index: int = 1):
        """Init."""
        super().__init__()
        self.encoder = Encoder(d_inp=d_inp,
                               d_model=d_model,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               max_len=max_len,
                               attention_drop=attention_drop,
                               residual_drop=residual_drop,
                               pos_encoding=pos_encoding,
                               embedding_drop=embedding_drop)
        self.decoder = Decoder(d_out=d_out,
                               d_model=d_model,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               d_ff=d_ff,
                               max_len=max_len,
                               attention_drop=attention_drop,
                               residual_drop=residual_drop,
                               pos_encoding=pos_encoding,
                               embedding_drop=embedding_drop)

        self.pad_index = pad_index

    def make_pad_mask(self, x: torch.Tensor):
        """Make padding mask.

        return the boolean mask (1: pad, 0: non-pad token)
        """
        # x: (batch) x (# tokens) -> (batch) x 1 x 1 x (# tokens) for broadcasting
        pad_mask = (x != self.pad_index).unsqueeze(1).unsqueeze(2)
        return pad_mask

    def make_dec_mask(self, x: torch.Tensor):
        """Make a mask for a decoder to don't cheat next tokens."""
        num_token = x.shape[1]

        # mask to don't care padding tokens.
        pad_mask = self.make_pad_mask(x)

        # masking for subsequent tokens
        sub_mask = torch.tril(torch.ones((num_token, num_token))).bool().to(x.device)

        dec_mask = pad_mask & sub_mask

        return dec_mask

    def forward(self, src: torch.Tensor, dst: torch.Tensor):
        """Forward."""
        src_pad_mask = self.make_pad_mask(src)
        dst_mask = self.make_dec_mask(dst)

        enc_src = self.encoder(src, src_pad_mask)

        out, attention = self.decoder(dst, enc_src, dst_mask, src_pad_mask)
        return out, attention


def initialize_weights_xavier(m):
    """Initialize a transformer."""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def initialize_weights_kaiming(m):
    """Initialize a transformer."""
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def count_parameters(model):
    """Count number of model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_transformer(args, d_inp, d_out):
    """Get a transformer."""
    model = Transformer(d_inp=d_inp,
                        d_out=d_out,
                        d_model=args.d_model,
                        d_ff=args.d_ff,
                        n_layers=args.n_layers,
                        n_heads=args.n_heads,
                        attention_drop=args.attention_drop,
                        residual_drop=args.residual_drop,
                        embedding_drop=args.embedding_drop,
                        max_len=args.max_len,
                        pos_encoding=args.pos_encoding,
                        pad_index=args.pad_index)

    if args.init == 'kaiming':
        model.apply(initialize_weights_kaiming)

    elif args.init == 'xavier':
        model.apply(initialize_weights_xavier)

    else:
        raise NotImplementedError(f"Not expected init alogirhtm: {args.init}")

    num_params = count_parameters(model)
    logging.info(f"Number of parameters: {num_params:,}")
    return model


if __name__ == '__main__':
    transformer = Transformer(d_inp=100,
                              d_out=300,
                              d_model=512,
                              d_ff=2048,
                              n_layers=6,
                              n_heads=8,
                              attention_drop=0.1,
                              residual_drop=0.1,
                              embedding_drop=0.1,
                              max_len=100,
                              pad_index=1)

    print((f"The model has {count_parameters(transformer):,} "
           "trainable parameters"))
