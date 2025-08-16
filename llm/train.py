from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from llm.tokenizer_word import WordTokenizer
from llm.model import MiniGPT

BLOCK_SIZE = 64
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10

# 1) コーパス読み込み
corpus = Path("data/corpus.txt").read_text(encoding="utf-8")

# 2) Tokenizer 作成
tok = WordTokenizer.build_from_text(corpus)

# 3) ids に変換
ids = tok.encode(corpus, add_special=False)

class TextDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)
    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.block_size])
        y = torch.tensor(self.ids[idx+1:idx+self.block_size+1])
        return x, y

ds = TextDataset(ids, BLOCK_SIZE)

n_samples = len(ds)
if n_samples == 0:
    raise ValueError(f"dataset too small: tokens={len(ids)}, BLOCK_SIZE={BLOCK_SIZE}")

batch_size = min(BATCH_SIZE, n_samples)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

print(f"[info] tokens={len(ids)} samples={n_samples} batch_size={batch_size}")

# 4) モデル初期化
model = MiniGPT(vocab_size=len(tok.id2tok), block_size=BLOCK_SIZE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# 5) 学習ループ
for ep in range(EPOCHS):
    total = 0.0
    steps = 0
    for x, y in dl:
        logits, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        steps += 1
    print(f"epoch={ep+1} loss={total/max(1, steps):.4f}")

# 6) 保存
torch.save(model.state_dict(), "ckpts/minigpt.pt")
tok.save("ckpts/tokenizer.json")
