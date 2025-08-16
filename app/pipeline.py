# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from rag.index import SimpleIndex

STOP_WORDS = {"どうやって", "何を", "手順", "ですか", "方法", "教えて", "してください", "とは"}

def _is_jp_char(ch: str) -> bool:
    # ざっくり日本語（漢字・ひらがな・カタカナ・長音・中点）
    code = ord(ch)
    return (
        0x3040 <= code <= 0x309F  # ひらがな
        or 0x30A0 <= code <= 0x30FF  # カタカナ
        or 0x4E00 <= code <= 0x9FFF  # CJK 漢字
        or ch in "ー・"              # よく使う記号
    )

def _jp_ngrams(text: str, n_low: int = 2, n_high: int = 4) -> List[str]:
    # 日本語部分だけを対象に 2〜4 文字の部分文字列を作る
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
    # 句読点などをスペース化し、粗いトークンを作る
    for ch in "？?！!。、，,.　\t\n":
        q = q.replace(ch, " ")
    rough = [t.strip() for t in q.split(" ") if t.strip()]
    rough = [t for t in rough if t not in STOP_WORDS]

    # 英数字はそのまま保持（VPN/Outlook/Duoなどに効く）
    alnums = [t for t in rough if any(c.isalnum() for c in t)]

    # 日本語 n-gram を追加（接続/認証/勤怠/メール などを拾う）
    grams = _jp_ngrams(q, 2, 4)

    # 重複除去の順序保持
    seen = set()
    keys: List[str] = []
    for t in alnums + grams:
        if t and t not in seen:
            seen.add(t)
            keys.append(t)
    return keys

def _pick_bullets_by_keywords(question: str, contexts: List[str]) -> List[str]:
    keys = _keywords_from_question(question)
    bullets: List[str] = []

    # 行リスト化（ノイズ除去）
    ctx_lines: List[List[str]] = []
    for c in contexts:
        lines = []
        for ln in c.splitlines():
            s = ln.strip().lstrip("・-").strip()
            if not s:
                continue
            if s.startswith("テスト用") or "#token-" in s:
                continue
            lines.append(s)
        if lines:
            ctx_lines.append(lines)

    # 1) まずはキーワード一致で抽出（順序保持）
    all_lines = [s for lines in ctx_lines for s in lines]
    for s in all_lines:
        if any(k in s for k in keys):
            if s not in bullets:
                bullets.append(s)

    # 2) ヒットが少ない場合は、最上位文書の先頭から補完（文脈を自然に）
    MIN_BULLETS = 3
    if len(bullets) < MIN_BULLETS and ctx_lines:
        for s in ctx_lines[0]:  # 上位1文書の順序を尊重
            if s not in bullets:
                bullets.append(s)
            if len(bullets) >= MIN_BULLETS:
                break

    return bullets


def _format_answer(bullets: List[str], sources: List[str]) -> str:
    if not bullets:
        src = "、".join(dict.fromkeys(sources)) if sources else "Doc1"
        return "■ 回答\n- わかりません\n■ 出典: " + src
    body = "■ 回答\n" + "\n".join(f"- {b}" for b in bullets)
    src = "、".join(dict.fromkeys(sources)) if sources else "Doc1"
    return f"{body}\n■ 出典: {src}"

def _src_name(index: SimpleIndex, i: int) -> str:
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
    index: SimpleIndex
    thresh: float = 0.12    # ← 少し緩め
    max_k: int = 5

    def __init__(self, ckpt_dir: str = "ckpts", thresh: float = 0.12, max_k: int = 5):
        self.index = SimpleIndex.load(ckpt_dir)
        self.thresh = thresh
        self.max_k = max_k

    def ask(self, question: str, top_k: int = 3) -> dict:
        hits: List[Tuple[int, float, str]] = self.index.query(question, top_k=top_k)
        filtered = [(i, s, t) for (i, s, t) in hits if s >= self.thresh]
        if not filtered:
            filtered = hits[:1]

        contexts = [t for (_, _, t) in filtered]
        bullets = _pick_bullets_by_keywords(question, contexts)

        if self.max_k and len(bullets) > self.max_k:
            bullets = bullets[: self.max_k]

        source_names = [_src_name(self.index, i) for (i, _, _) in filtered]
        answer = _format_answer(bullets, source_names)

        return {
            "query": question,
            "contexts": [
                {"rank": r + 1, "score": float(s), "text": t}
                for r, (i, s, t) in enumerate(filtered)
            ],
            "answer": answer,
        }
