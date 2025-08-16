from textwrap import dedent

def build_prompt(question: str, contexts: list[str]):
    ctx = "\n\n".join(f"[Doc{i+1}]\n{c}" for i, c in enumerate(contexts))
    system = dedent("""
        あなたは社内ヘルプデスクのアシスタントです。
        以下のドキュメントのみを根拠に、箇条書きで簡潔に回答してください。
        不明な点は「わかりません」と書いてください。
        最後に出典として [Doc1] のように参照を付けてください。
        出力は次の形式に厳密に従ってください。

        回答:
        - ...
        - ...
        出典: [Doc?]
    """).strip()
    user = f"質問: {question}\n\nコンテキスト:\n{ctx}\n\n出力のみを書いてください。"
    return system + "\n\n" + user
