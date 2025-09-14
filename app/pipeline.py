# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rag.index import SimpleIndex

# ------------------------------------------------------------
# 検索クエリ（日本語の質問）からノイズになりがちな語を除外するための簡易ストップワード。
# 例: 「〜の手順を教えてください」「〜とは？」などで、意味的なキーワード抽出を邪魔する語。
# 必要に応じてドメイン（社内用語など）に合わせて調整・拡張する。
# ------------------------------------------------------------
STOP_WORDS = {"どうやって", "何を", "手順", "ですか", "方法", "教えて", "してください", "とは"}

def _is_jp_char(ch: str) -> bool:
    """
    ざっくり「日本語っぽい1文字」かどうかを判定するヘルパー。
    - 対象: ひらがな / カタカナ / CJK 漢字 / よく使う記号（長音'ー'・中点'・'）
    - 目的: 英数や記号に混じる日本語部分のみを n-gram 抽出の対象にするため。
    """
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F  # ひらがな
        or 0x30A0 <= code <= 0x30FF  # カタカナ
        or 0x4E00 <= code <= 0x9FFF  # CJK 漢字
        or ch in "ー・"              # よく使う日本語記号
    )

def _jp_ngrams(text: str, n_low: int = 2, n_high: int = 4) -> List[str]:
    """
    入力文字列から「日本語部分のみ」を対象に 2〜4 文字の部分文字列（文字 n-gram）を作る。
    - 手順:
      1) 日本語文字のみ残し、非日本語はスペースへ（単語境界を作る）
      2) スペースで分割して「日本語の塊」を得る
      3) 各塊から n ∈ [n_low, n_high] の連続部分列を列挙
    - 目的: 形態素解析を使わずとも日本語の“キーワード断片”を拾うための簡便策。
    """
    s = "".join(ch if _is_jp_char(ch) else " " for ch in text)
    tokens = [t for t in s.split() if t]
    grams: List[str] = []
    for t in tokens:
        L = len(t)
        for n in range(n_low, n_high + 1):
            for i in range(0, max(0, L - n + 1)):
                grams.append(t[i:i+n])
    return grams

def _keywords_from_question(q: str) -> List[str]:
    """
    質問文 q から検索に使うキーワード候補を抽出する。
    - 句読点/記号 → スペース化 → 粗いトークン列（rough）を作る
    - ストップワードを除外
    - 英数字を含むトークン（VPN, Outlook, Duo などの製品名）を優先的に保持
    - さらに日本語の 2〜4 文字 n-gram を追加
    - 最後に順序を保った重複除去でまとめる
    """
    # 句読点・全角記号などを空白に。split で粗いトークン化。
    for ch in "？?！!。、，,.　\t\n":
        q = q.replace(ch, " ")
    rough = [t.strip() for t in q.split(" ") if t.strip()]
    # 汎用語（ストップワード）を間引く
    rough = [t for t in rough if t not in STOP_WORDS]

    # 英数を含むトークン（製品名・略語・型番などに効く）
    alnums = [t for t in rough if any(c.isalnum() for c in t)]

    # 日本語の主要キーワードを拾うため、n-gram で網を広げる
    grams = _jp_ngrams(q, 2, 4)

    # 順序保持の重複除去（先に alnums、次に grams を優先度順で連結）
    seen = set()
    keys: List[str] = []
    for t in alnums + grams:
        if t and t not in seen:
            seen.add(t)
            keys.append(t)
    return keys

def _pick_bullets_by_keywords(question: str, contexts: List[str]) -> List[str]:
    """
    上位文書の本文（contexts）から、質問に関係しそうな箇条書き（bullets）を抽出する。
    手順:
      1) contexts を行単位に分割し、ノイズ行を除去（空行・「テスト用」・token行など）
      2) 質問から抽出した keywords が含まれる行を順序保持で収集
      3) 抽出が少ない場合は、最上位文書の先頭から補完して最低数を満たす
    目的:
      - TF-IDF の上位文書は合っていても、本文は長いことが多い。
        キーワードと行単位のヒューリスティクスで“答えになりそうな箇所”をつまむ。
    """
    keys = _keywords_from_question(question)
    bullets: List[str] = []

    # 行リスト化（ノイズ・メタ行の除去）
    ctx_lines: List[List[str]] = []
    for c in contexts:
        lines = []
        for ln in c.splitlines():
            # 行頭の「・」「-」などの箇条書き記号を剥がして余白を整える
            s = ln.strip().lstrip("・-").strip()
            if not s:
                continue
            # 明示的に無視したい行のルール
            if s.startswith("テスト用") or "#token-" in s:
                continue
            lines.append(s)
        if lines:
            ctx_lines.append(lines)

    # 1) キーワード一致行を順序保持で抽出
    all_lines = [s for lines in ctx_lines for s in lines]
    for s in all_lines:
        if any(k in s for k in keys):
            if s not in bullets:
                bullets.append(s)

    # 2) 最低件数に満たない場合、最上位文書の先頭から補完（文脈の自然さを重視）
    MIN_BULLETS = 3
    if len(bullets) < MIN_BULLETS and ctx_lines:
        for s in ctx_lines[0]:  # 最上位文書の順序を尊重
            if s not in bullets:
                bullets.append(s)
            if len(bullets) >= MIN_BULLETS:
                break

    return bullets

def _format_answer(bullets: List[str], sources: List[str]) -> str:
    """
    箇条書き bullets と出典 sources から最終回答のテキストを整形する。
    - bullets が空なら「わかりません」を返す（出典はあれば併記）
    - sources は重複除去＆順序保持で "、" 連結
    出力例:
      ■ 回答
      - xxx
      - yyy
      ■ 出典: 001_入館とVPN, 002_勤怠と休暇
    """
    if not bullets:
        src = "、".join(dict.fromkeys(sources)) if sources else "Doc1"
        return "■ 回答\n- わかりません\n■ 出典: " + src
    body = "■ 回答\n" + "\n".join(f"- {b}" for b in bullets)
    src = "、".join(dict.fromkeys(sources)) if sources else "Doc1"
    return f"{body}\n■ 出典: {src}"

def _src_name(index: SimpleIndex, i: int) -> str:
    """
    SimpleIndex から i 番目ドキュメントの「出典名」を安全に取得する。
    - 優先: index.source_name(i) が提供されていればそれを使う
    - 次点: index.paths[i] の stem（拡張子なしファイル名）
    - 最後: "Doc{i}" フォールバック
    例外・属性未定義に対して頑健に作っている。
    """
    if hasattr(index, "source_name"):
        try:
            return index.source_name(i)  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(index, "paths") and getattr(index, "paths"):
        try:
            return Path(index.paths[i]).stem  # type: ignore[attr-defined]
        except Exception:
            pass
    return f"Doc{i}"

@dataclass
class QAPipeline:
    """
    「最小 RAG」用の問い合わせパイプライン。
    - 検索: SimpleIndex (TF-IDF + cos 類似) で上位文書を取得
    - 抽出: キーワードベースで行抽出して箇条書きを生成
    - 整形: 回答テキストにまとめ、出典も併記
    - パラメータ:
        thresh: 文書スコア（cos 類似）の下限しきい値。低すぎるヒットを弾く。
        max_k:  箇条書きの上限件数。冗長な回答を避ける。
    """
    index: SimpleIndex
    thresh: float = 0.12    # ← 少し緩め（スコアが低い環境や短文でも最低限拾うため）
    max_k: int = 5

    def __init__(self, ckpt_dir: str = "ckpts", thresh: float = 0.12, max_k: int = 5):
        # ckpts/ から TF-IDF 行列・メタ・ベクトライザをロードして検索器を準備
        self.index = SimpleIndex.load(ckpt_dir)
        self.thresh = thresh
        self.max_k = max_k

    def ask(self, question: str, top_k: int = 3) -> dict:
        """
        ユーザの質問に対して、検索→抽出→整形を行い、回答オブジェクトを返す。
        返り値の構造（例）:
        {
          "query": "社外からVPNに接続する手順は？",
          "contexts": [
            {"rank": 1, "score": 0.83, "text": "..."},
            {"rank": 2, "score": 0.61, "text": "..."},
            ...
          ],
          "answer": "■ 回答\n- ...\n- ...\n■ 出典: 001_入館とVPN"
        }
        """
        # 1) TF-IDF 検索（コサイン類似度の降順が返る想定）
        hits: List[Tuple[int, float, str]] = self.index.query(question, top_k=top_k)

        # 2) スコアしきい値でフィルタ（0.12 未満は弱ヒットとして除外）
        #    ただし何も残らなければ最上位1件だけは採用して「完全無回答」を避ける。
        filtered = [(i, s, t) for (i, s, t) in hits if s >= self.thresh]
        if not filtered:
            filtered = hits[:1]

        # 3) 行抽出用に本文だけ取り出す → キーワードに基づき箇条書きを生成
        contexts = [t for (_, _, t) in filtered]
        bullets = _pick_bullets_by_keywords(question, contexts)

        # 4) 箇条書きの件数上限（冗長防止）
        if self.max_k and len(bullets) > self.max_k:
            bullets = bullets[: self.max_k]

        # 5) 出典名の整形（重複を避けるのは _format_answer 側で実施）
        source_names = [_src_name(self.index, i) for (i, _, _) in filtered]
        answer = _format_answer(bullets, source_names)

        # 6) API レスポンス用の構造化（コンテキストは rank/score/text を返す）
        return {
            "query": question,
            "contexts": [
                {"rank": r + 1, "score": float(s), "text": t}
                for r, (i, s, t) in enumerate(filtered)
            ],
            "answer": answer,
        }
