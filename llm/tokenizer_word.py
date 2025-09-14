# tokenizer_word.py (simplified)
# -*- coding: utf-8 -*-
from pathlib import Path
import json
from collections import Counter
from fugashi import Tagger

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

class WordTokenizer:
    """fugashi で分かち書きし、表層形ベースで語彙管理する最小実装。"""
    _tagger = Tagger()  # 1プロセスで1回だけ初期化

    def __init__(self, vocab):
        self.id2tok = list(vocab)
        self.tok2id = {t: i for i, t in enumerate(self.id2tok)}
        # 便利ID
        self.pad_id = self.tok2id["<PAD>"]
        self.bos_id = self.tok2id["<BOS>"]
        self.eos_id = self.tok2id["<EOS>"]
        self.unk_id = self.tok2id["<UNK>"]

    # ---- 内部ユーティリティ ----
    @staticmethod
    def _wakati_text(text: str):
        return [m.surface for m in WordTokenizer._tagger(text)]

    # ---- ビルド ----
    @classmethod
    def build_from_text(cls, text: str, min_freq: int = 1, max_vocab: int | None = None):
        freqs = Counter(cls._wakati_text(text))
        # 最低頻度でフィルタ
        items = [(w, c) for w, c in freqs.items() if c >= min_freq]
        # 出現頻度降順・語彙順で安定ソート
        items.sort(key=lambda x: (-x[1], x[0]))
        # 語彙上限（特殊トークンぶん確保）
        if max_vocab is not None:
            if max_vocab < len(SPECIALS):
                raise ValueError("max_vocab must be >= len(SPECIALS)")
            items = items[: max_vocab - len(SPECIALS)]
        vocab = SPECIALS + [w for w, _ in items]
        return cls(vocab)

    # ---- 変換 ----
    def encode(self, s: str, add_special: bool = True):
        ids = []
        if add_special:
            ids.append(self.bos_id)
        for w in self._wakati_text(s):
            ids.append(self.tok2id.get(w, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        toks = [self.id2tok[i] for i in ids if i not in specials]
        return "".join(toks)  # 日本語は空白なし

    # ---- 永続化 ----
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.id2tok}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str):
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))["vocab"]
        return WordTokenizer(vocab)

# 単体実行で簡易ビルド
if __name__ == "__main__":
    txt = Path("data/corpus.txt").read_text(encoding="utf-8")
    tok = WordTokenizer.build_from_text(txt, min_freq=1, max_vocab=20000)
    tok.save("ckpts/tokenizer.json")
    print("saved tokenizer.json | vocab_size=", len(tok.id2tok))
