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
        シンプルな TF-IDF ベースの検索インデックス。
        - 役割: クエリ文字列を訓練時と同じベクトライザで TF-IDF 化し、
                文書行列 X とのコサイン類似度で上位文書を返す。

        Parameters
        ----------
        X :
            shape=(n_docs, vocab) の文書行列（TF-IDF）。各行は **L2正規化済み** を想定。
            dtype は float32 推奨（省メモリ・十分な精度）。
        texts :
            各ドキュメントの本文（またはスニペット）。検索結果に一緒に返す。
        vec :
            学習時のベクトライザ（例: sklearn.feature_extraction.text.TfidfVectorizer）。
            クエリを同じ前処理・語彙で数値化するために必須。
        paths :
            各ドキュメントのファイルパス（UIで出典を表示する用途）。省略可。
        """
        # X は float32 に統一（copy=False なので既に float32 ならコピーしない）
        self.X = X.astype(np.float32, copy=False)
        self.texts = texts
        self.vec = vec
        self.paths = paths or []

    @staticmethod
    def load(ckpt_dir: str | Path) -> "SimpleIndex":
        """
        build_index.py で保存した成果物（ckpts/配下）からインデックスを復元する。

        期待ファイル:
          - tfidf.npz                ... {"X": (n_docs, vocab)} の Numpy アーカイブ
          - tfidf_meta.json          ... {"texts": [...], "vec_path": "...", "paths": [...]}
          - tfidf_vectorizer.joblib  ... sklearn の TfidfVectorizer を joblib で保存したもの
        """
        ckpt_dir = Path(ckpt_dir)

        # 1) ベクトル本体のロード
        data = np.load(ckpt_dir / "tfidf.npz")
        X = data["X"]  # shape=(n_docs, vocab), L2正規化済み想定

        # 2) メタ情報（原文やファイルパス、ベクトライザパス）
        meta = json.loads((ckpt_dir / "tfidf_meta.json").read_text(encoding="utf-8"))
        texts = meta["texts"]
        vec_path = meta["vec_path"]
        paths = meta.get("paths", [])  # paths が無い保存形式にも耐性

        # 3) ベクトライザ（クエリ側で再利用）
        vec = load(vec_path)

        return SimpleIndex(X=X, texts=texts, vec=vec, paths=paths)

    def source_name(self, i: int) -> str:
        """
        検索結果で表示する出典名（ファイル名の stem）。
        paths を持っていない場合は "Doc{番号}" にフォールバック。
        """
        if self.paths and 0 <= i < len(self.paths):
            return Path(self.paths[i]).stem
        return f"Doc{i+1}"

    def query(self, question: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        クエリ文字列に対する上位文書を返す。

        Returns
        -------
        List[Tuple[doc_index, score, text]]
            スコア（コサイン類似度）の降順に並んだタプルのリスト。
            - doc_index : ヒットした文書のインデックス（texts/paths へ参照可能）
            - score     : 類似度（-1.0〜1.0。ここでは TF-IDF なので 0〜1.0 付近が多い）
            - text      : 対応する本文（スニペット）

        NOTE
        ----
        - X はすでに L2 正規化済みを前提とし、クエリ側だけここで正規化する。
        - ベクトライザ vec は build_index 時のものと一致している必要がある。
        """
        # 1) クエリを TF-IDF 化（疎行列 → dense 1D ベクトルへ）
        #    .toarray() で (1, vocab) → [vocab] に変形し float32 化
        qv = self.vec.transform([question]).toarray().astype(np.float32)[0]  # shape=(vocab,)

        # 2) クエリの L2 正規化（ゼロ割り防止の微小量付き）
        denom = float(np.linalg.norm(qv) + 1e-12)
        qn = qv / denom  # shape=(vocab,)

        # 3) コサイン類似度の計算
        #    文書側は L2 正規化済み → cos(q, d) = (qn・d) = 行列積で OK
        #    scores: shape=(n_docs,)
        scores = self.X @ qn

        # 4) 上位 top_k を抽出（降順）
        #    np.argsort は O(n log n)。n_docs が非常に大きい場合は argpartition の検討余地あり。
        top_k = max(1, int(top_k))  # 不正値に対する最低1件ガード
        idx = np.argsort(-scores)[:top_k]

        # 5) 結果の整形（必要最小限: index, score, text）
        out: List[Tuple[int, float, str]] = [
            (int(i), float(scores[i]), self.texts[i]) for i in idx
        ]
        return out
