# llm/generate.py
# -*- coding: utf-8 -*-
import torch
from llm.model import MiniGPT
from llm.tokenizer_word import WordTokenizer

def _top_k_filter(logits, k=40):
    if k is None or k <= 0: return logits
    v, _ = torch.topk(logits, k)
    thr = v[..., -1, None]
    return torch.where(logits < thr, torch.full_like(logits, -1e10), logits)

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=80, temperature=0.5,
                  repetition_penalty=1.2, top_k=40, no_repeat_ngram=3):
    tok = WordTokenizer.load("ckpts/tokenizer.json")
    ckpt = torch.load("ckpts/llm.pt", map_location="cpu")

    model = MiniGPT(vocab_size=ckpt["vocab_size"], block_size=ckpt["block_size"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    id_eos = tok.tok2id.get("<EOS>")

    def violates(ids, cand, n):
        if n <= 1 or ids.size(1) < n-1: return False
        seq = ids[0].tolist()
        pat = seq[-(n-1):] + [int(cand)]
        for i in range(len(seq)-n+1):
            if seq[i:i+n] == pat: return True
        return False

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :]

        # repetition penalty（出た単語はやや不利に）
        for t in set(idx[0].tolist()):
            logits[0, t] /= repetition_penalty

        logits = _top_k_filter(logits, top_k)

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            tries = 0
            while True:
                next_id = torch.multinomial(probs, num_samples=1)
                if not violates(idx, next_id[0,0], no_repeat_ngram) or tries > 10:
                    break
                probs[0, next_id] = 0
                probs = probs / probs.sum()
                tries += 1
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if id_eos is not None and int(next_id) == id_eos:
            break
        idx = torch.cat([idx, next_id], dim=1)

        # 行儀よく終了（簡易）
        out_txt = tok.decode(idx[0].tolist())
        if out_txt.endswith("\n\n"):
            break

    return tok.decode(idx[0].tolist())
