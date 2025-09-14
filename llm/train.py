# train.py
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from llm.tokenizer_word import WordTokenizer
from llm.model import MiniGPT

# ---- ハイパラ ----
BLOCK_SIZE = 64
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
torch.manual_seed(42)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

# ---- データ読み込み & トークナイズ ----
corpus = Path("data/corpus.txt").read_text(encoding="utf-8")
tok = WordTokenizer.build_from_text(corpus)
ids = tok.encode(corpus, add_special=False)  # [t0, t1, ..., tN]

# ---- Dataset ----
class TextDataset(Dataset):
    """スライディングウィンドウで (x, y) を作る固定長データセット"""
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        # 作れるサンプル数（y が1つ先まで必要なので -1）
        self.n = max(0, len(ids) - block_size - 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + self.block_size + 1], dtype=torch.long)
        return x, y

ds = TextDataset(ids, BLOCK_SIZE)

if len(ds) == 0:
    raise ValueError(f"dataset too small: tokens={len(ids)}, BLOCK_SIZE={BLOCK_SIZE}")

batch_size = min(BATCH_SIZE, len(ds))

dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,   # 小規模/デバッグ時は 0 が扱いやすい
)

print(f"[info] tokens={len(ids)} samples={len(ds)} batch_size={batch_size} device={device}")

# ---- モデル & 最適化 ----
model = MiniGPT(vocab_size=len(tok.id2tok), block_size=BLOCK_SIZE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ---- 学習ループ ----
model.train()
for ep in range(1, EPOCHS + 1):
    total = 0.0
    steps = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 必要ならON
        opt.step()
        total += loss.item()
        steps += 1
    print(f"epoch={ep} loss={total / max(1, steps):.4f}")

# ---- 保存 ----
Path("ckpts").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "ckpts/minigpt.pt")
tok.save("ckpts/tokenizer.json")
