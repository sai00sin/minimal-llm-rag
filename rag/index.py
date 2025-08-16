# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from joblib import load


class SimpleIndex:
    def __init__(self, X: np.ndarray, texts: List[str], vec, paths: List[str] | None = None):
        """
        X: shape (n_docs, vocab), L2 正規化済みの dense np.ndarray(float32 推奨)
        texts: 各ドキュメントの本文（またはスニペット）
        vec: 学習時のベクトライザ（sklearn の TfidfVectorizer など）
        paths: 各ドキュメントのファイルパス（出典表示用、任意）
        """
        self.X = X.astype(np.float32, copy=False)
        self.texts = texts
        self.vec = vec
        self.paths = paths or []

    @staticmethod
    def load(ckpt_dir: str | Path) -> "SimpleIndex":
        ckpt_dir = Path(ckpt_dir)
        # ベクトル本体
        data = np.load(ckpt_dir / "tfidf.npz")
        X = data["X"]

        # メタ
        meta = json.loads((ckpt_dir / "tfidf_meta.json").read_text(encoding="utf-8"))
        texts = meta["texts"]
        vec_path = meta["vec_path"]
        paths = meta.get("paths", [])  # ない場合もある

        # ベクトライザ
        vec = load(vec_path)

        return SimpleIndex(X=X, texts=texts, vec=vec, paths=paths)

    def source_name(self, i: int) -> str:
        """出典名（ファイル名 stem）。paths がなければ Doc番号 にフォールバック。"""
        if self.paths and 0 <= i < len(self.paths):
            return Path(self.paths[i]).stem
        return f"Doc{i+1}"

    def query(self, question: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        返り値: [(doc_index, score, text), ...] スコア降順
        """
        # 1) 疎行列 → dense 1D に変換
        qv = self.vec.transform([question]).toarray().astype(np.float32)[0]  # shape (vocab,)

        # 2) L2 正規化（ゼロ割り防止）
        denom = float(np.linalg.norm(qv) + 1e-12)
        qn = qv / denom  # shape (vocab,)

        # 3) コサイン（X は行ベクトルがすでに L2 正規化済みを想定）
        scores = self.X @ qn  # shape (n_docs,)

        # 4) 上位 top_k 抽出
        top_k = max(1, int(top_k))
        idx = np.argsort(-scores)[:top_k]
        out: List[Tuple[int, float, str]] = [(int(i), float(scores[i]), self.texts[i]) for i in idx]
        return out
