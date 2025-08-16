# rag/vectorizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from typing import List

# ←← これが「学習時もロード時も」同じファイル・同じ名前で存在すること！
def _wakati(text: str) -> List[str]:
    # すごく簡易な日本語分かち（必要ならMeCab等に差し替え可）
    text = re.sub(r"[^\w一-龥ぁ-んァ-ヶｦ-ﾟ]+", " ", text)
    toks = [t for t in text.split() if t]
    return toks

def _preprocess(s: str) -> str:
    # ごく軽い正規化（全角空白→半角、改行→空白）
    return s.replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")

def build_tfidf(docs: List[str]) -> Tuple[np.ndarray, list[str], TfidfVectorizer]:
    """
    日本語向けに文字 n-gram ベースの TF-IDF ベクトルを作る。
    - analyzer='char_wb'（単語境界を考慮した文字 n-gram）
    - ngram_range=(2,4) あたりが日本語では扱いやすい
    """
    docs_norm = [_preprocess(d) for d in docs]
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
        min_df=1,
        max_df=1.0,
        norm="l2",
        use_idf=True,
        sublinear_tf=True,
    )
    X = vec.fit_transform(docs_norm).astype(np.float32).toarray()
    vocab = list(vec.get_feature_names_out())
    return X, vocab, vec

def encode_tfidf(vec: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    texts_norm = [_preprocess(t) for t in texts]
    return vec.transform(texts_norm).astype(np.float32).toarray()
