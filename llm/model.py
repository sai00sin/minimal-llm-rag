# llm/model_simple.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    """Pre-LN: x = x + Attn(LN(x)); x = x + MLP(LN(x))"""
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd, num_heads=n_head, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.ln1(x)
        T = x1.size(1)

        # 上三角（未来）を True でマスクするブーリアンマスク (T,T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # 明示的に attn_mask を渡す（is_causal は不要なので外してOK）
        y, _ = self.attn(x1, x1, x1, need_weights=False, attn_mask=causal_mask)

        x = x + self.drop(y)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """最小 GPT 風モデル（トークン埋め込み + 位置埋め込み + Transformer ブロック列）"""
    def __init__(self, vocab_size, n_layer=2, n_head=2, n_embd=128, block_size=128,
                 dropout=0.1, tie_weights=True):
        super().__init__()
        self.block_size = block_size

        # 埋め込み
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)

        # ブロック
        self.blocks = nn.ModuleList([GPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)

        # 出力（語彙次元へ）
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # 重み共有（パラメータ削減＆性能向上の定番）
        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        """
        idx: (B, T), targets: (B, T) or None
        returns: (logits: (B, T, V), loss or None)
        """
        B, T = idx.size()
        if T > self.block_size:
            raise ValueError(f"T={T} > block_size={self.block_size}")

        pos = torch.arange(T, device=idx.device)  # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]  # (B,T,C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss
