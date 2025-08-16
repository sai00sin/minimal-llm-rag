# rag/build_index.py
# -*- coding: utf-8 -*-
from pathlib import Path
from joblib import dump
import numpy as np
from sklearn.preprocessing import normalize
from rag.vectorizer import build_tfidf
from rag.index import SimpleIndex

def read_docs(d):
    paths = sorted(Path(d).glob("*.txt"))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return texts, paths

if __name__ == "__main__":
    texts, paths = read_docs("data/faq_docs")
    X, vocab, vec = build_tfidf(texts)

    # cos類似用に L2 正規化
    X = normalize(X, axis=1).astype(np.float32)

    Path("ckpts").mkdir(parents=True, exist_ok=True)
    np.savez("ckpts/tfidf.npz", X=X)

    meta = {
        "texts": texts,
        "paths": [str(p) for p in paths],       # ← これを保存
        "vec_path": "ckpts/tfidf_vectorizer.joblib",
    }
    (Path("ckpts")/"tfidf_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    dump(vec, "ckpts/tfidf_vectorizer.joblib")

    # 早期検証（ロードできるか）
    _ = SimpleIndex.load("ckpts")
    print("saved ckpts/tfidf.npz, tfidf_meta.json, tfidf_vectorizer.joblib | docs=", len(texts))
