# rag/build_index.py
# -*- coding: utf-8 -*-
from pathlib import Path
from joblib import dump
import numpy as np
from sklearn.preprocessing import normalize
from rag.vectorizer import build_tfidf
from rag.index import SimpleIndex

def read_docs(d):
    """
    指定ディレクトリ d 配下の .txt を読み込み、テキストと Path のリストを返す。
    - 返り値:
        texts: [str, ...]   各ファイルの本文
        paths: [Path, ...]  対応するファイルパス（後でメタ表示・デバッグに使う）
    """
    # 再現性のため glob 結果はソート（OS依存の順序ゆらぎを抑える）
    paths = sorted(Path(d).glob("*.txt"))
    # UTF-8 で素直に読む（社内Q&A前提のため BOM 等は想定しない）
    texts = [p.read_text(encoding="utf-8") for p in paths]
    return texts, paths

if __name__ == "__main__":
    # ------------------------------------------------------------
    # 1) 文書を読み込む
    # ------------------------------------------------------------
    # data/faq_docs/ に社内Q&Aの原稿 .txt を並べる想定
    texts, paths = read_docs("data/faq_docs")

    # ------------------------------------------------------------
    # 2) TF-IDF ベクトル化
    # ------------------------------------------------------------
    # - build_tfidf(texts) は以下を返す想定:
    #     X:     (N_docs, V) の TF-IDF 行列（numpy または scipy → ここでは toarray 済）
    #     vocab: 語彙（使わないが、解析・可視化時に役立つ）
    #     vec:   学習済み TfidfVectorizer（検索時のクエリを同じ前処理でベクトル化するため必須）
    X, vocab, vec = build_tfidf(texts)

    # ------------------------------------------------------------
    # 3) cos 類似の前処理として L2 正規化
    # ------------------------------------------------------------
    # - コサイン類似度 sim(q, d) = (q・d) / (||q|| * ||d||)
    #   文書側 D を L2 正規化しておくと、検索時は
    #   sim(q, d) = (q_norm・d_norm) で単なる内積になる。
    # - dtype は float32 に落とし、ディスク占有とメモリを削減。
    X = normalize(X, axis=1).astype(np.float32)

    # ------------------------------------------------------------
    # 4) 生成物の保存（ckpts/ に集約）
    # ------------------------------------------------------------
    # - tfidf.npz:   文書ベクトル群（X）
    # - tfidf_meta.json: 検索時に原文やパスを引くためのメタ情報
    # - tfidf_vectorizer.joblib: クエリ側のベクトル化で再利用
    Path("ckpts").mkdir(parents=True, exist_ok=True)
    np.savez("ckpts/tfidf.npz", X=X)

    # メタ情報:
    # - texts: 検索ヒットをそのまま返す・抜粋生成のため
    # - paths: どのファイルだったかを UI ログ/リンク表示に使う
    # - vec_path: ベクトライザの保存先（SimpleIndex がロード時に参照する前提）
    meta = {
        "texts": texts,
        "paths": [str(p) for p in paths],       # 後でパスを文字列で復元しやすくする
        "vec_path": "ckpts/tfidf_vectorizer.joblib",
    }
    (Path("ckpts")/"tfidf_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Vectorizer 本体は joblib で保存（scikit-learn の標準手法）
    dump(vec, "ckpts/tfidf_vectorizer.joblib")

    # ------------------------------------------------------------
    # 5) 早期検証（フォーマット/パス不整合の早期発見）
    # ------------------------------------------------------------
    # - SimpleIndex.load("ckpts") が例外なく動けば、最低限の読込経路は保証される。
    # - ここでは戻り値は捨てる（存在確認のみ）。
    _ = SimpleIndex.load("ckpts")

    print("saved ckpts/tfidf.npz, tfidf_meta.json, tfidf_vectorizer.joblib | docs=", len(texts))
