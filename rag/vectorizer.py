# rag/vectorizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from typing import List

# ※重要※
# 「学習時に使った前処理・分かち方・パラメータ」と
# 「推論時（ロード後）に使うそれら」は 100% 一致している必要がある。
# ここで定義した関数名や設定値を変えると、検索品質が大きく落ちるので要注意。

def _wakati(text: str) -> List[str]:
    """
    （現在は未使用だが将来の差し替え用）超簡易な日本語分かち。
    - 記号等をスペースへ置換 → 空白で split するだけ。
    - MeCab などの形態素解析へ切り替える場合の“足場”として残している。
    """
    # \w（英数アンダースコア）と日本語の主要ブロック以外をスペースに
    text = re.sub(r"[^\w一-龥ぁ-んァ-ヶｦ-ﾟ]+", " ", text)
    toks = [t for t in text.split() if t]
    return toks

def _preprocess(s: str) -> str:
    """
    ごく軽い正規化（文字数を増やさず無害化する範囲に限定）
    - 全角空白→半角空白
    - 改行コードの正規化（\r\n / \r → \n）
    """
    return s.replace("\u3000", " ").replace("\r\n", "\n").replace("\r", "\n")

def build_tfidf(docs: List[str]) -> Tuple[np.ndarray, list[str], TfidfVectorizer]:
    """
    文書集合 docs を TF-IDF ベクトル化する（日本語向けに“文字 n-gram”ベース）。
    設計ポイント:
      - analyzer='char_wb':
          単語境界（空白）内で文字 n-gram を作る。日本語では空白が少ないが、
          記号→空白の前処理により、文の“塊”を保った n-gram が得られやすい。
      - ngram_range=(2,4):
          2〜4 文字の n-gram。単語分割なしでも程よく表現力が出るバランス設定。
      - sublinear_tf=True:
          tf を 1 + log(tf) に圧縮し、長文の影響を緩和。
      - norm='l2':
          ベクトルを L2 正規化（コサイン類似度前提）。後段で再度 normalize しても整合。
    戻り値:
      X:    (n_docs, vocab) の dense 行列（float32）
      vocab: 使用語彙（n-gram）のリスト
      vec:   学習済み TfidfVectorizer（推論時のクエリ変換に必須）
    """
    # 1) 軽い前処理をかけ、学習・推論で同じ関数を共有することが重要
    docs_norm = [_preprocess(d) for d in docs]

    # 2) TF-IDF ベクトライザを定義
    vec = TfidfVectorizer(
        analyzer="char_wb",     # 文字 n-gram（単語境界単位）
        ngram_range=(2, 4),     # 2〜4 文字
        lowercase=False,        # 日本語の大文字小文字変換は基本不要
        min_df=1,               # すべての n-gram を対象（小規模コーパス前提）
        max_df=1.0,             # 上限なし（大規模なら 0.9 などに調整可）
        norm="l2",              # L2 正規化（疎→密にする前に内部で正規化）
        use_idf=True,           # IDF を使用
        sublinear_tf=True,      # tf を対数圧縮
    )

    # 3) 学習 & 変換（疎→dense、float32 にダウンクォート）
    #   ※ toarray() は密行列化するため文書数/語彙数が大きいとメモリを食う。
    #     必要に応じて疎のまま扱う設計に変更する。
    X = vec.fit_transform(docs_norm).astype(np.float32).toarray()

    # 4) 語彙リスト（モデル可視化やデバッグで役立つ）
    vocab = list(vec.get_feature_names_out())
    return X, vocab, vec

def encode_tfidf(vec: TfidfVectorizer, texts: List[str]) -> np.ndarray:
    """
    既存の（学習済み）ベクトライザ vec を用いて texts を TF-IDF へ変換するユーティリティ。
    - build_tfidf と同じ _preprocess を通すことで、学習/推論の整合性を担保。
    - 返り値は dense の float32（toarray() はメモリに注意）。
    """
    texts_norm = [_preprocess(t) for t in texts]
    return vec.transform(texts_norm).astype(np.float32).toarray()
