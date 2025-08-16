# -*- coding: utf-8 -*-
from pathlib import Path
import json
from fugashi import Tagger

SPECIALS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

class WordTokenizer:
    """MeCab(fugashi)で分かち書き → 語彙化する最小実装。"""
    def __init__(self, vocab=None):
        self.tagger = Tagger()
        if vocab:
            self.id2tok = vocab
            self.tok2id = {t: i for i, t in enumerate(vocab)}
        else:
            self.id2tok, self.tok2id = None, None

    def _wakati(self, s: str):
        # 表層形ベースの最小分かち
        return [m.surface for m in self.tagger(s)]

    @staticmethod
    def build_from_text(text: str, min_freq: int = 1, max_vocab: int | None = None):
        t = WordTokenizer()
        freqs = {}
        for w in t._wakati(text):
            freqs[w] = freqs.get(w, 0) + 1
        # 出現頻度でフィルタ
        items = [(w, c) for w, c in freqs.items() if c >= min_freq]
        # 頻度降順→語順で安定化
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_vocab:
            items = items[:max_vocab - len(SPECIALS)]
        vocab = SPECIALS + [w for w, _ in items]
        tok = WordTokenizer(vocab)
        return tok

    def encode(self, s: str, add_special=True):
        ids = []
        if add_special:
            ids.append(self.tok2id["<BOS>"])
        for w in self._wakati(s):
            i = self.tok2id.get(w, self.tok2id["<UNK>"])
            ids.append(i)
        if add_special:
            ids.append(self.tok2id["<EOS>"])
        return ids

    def decode(self, ids):
        specials = {self.tok2id.get(x) for x in SPECIALS if self.tok2id and x in self.tok2id}
        toks = [self.id2tok[i] for i in ids if i not in specials]
        # 日本語なので空白連結はしない（必要なら" ".joinに変更）
        return "".join(toks)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.id2tok}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str):
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))["vocab"]
        return WordTokenizer(vocab)

if __name__ == "__main__":
    txt = Path("data/corpus.txt").read_text(encoding="utf-8")
    tok = WordTokenizer.build_from_text(txt, min_freq=1, max_vocab=20000)
    tok.save("ckpts/tokenizer.json")
    print("saved tokenizer.json | vocab_size=", len(tok.id2tok))
