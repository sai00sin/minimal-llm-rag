# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.pipeline import QAPipeline

# ============================================================
# FastAPI アプリ定義
# ------------------------------------------------------------
# - title: ドキュメントUI（/docs, /redoc）や OpenAPI スキーマの表記に使われる
# - 本APIは「最小の RAG（Retrieve & Generate）デモ」用
# ============================================================
app = FastAPI(title="minimal-llm-rag")

# ============================================================
# パイプラインの常駐インスタンス
# ------------------------------------------------------------
# - QAPipeline() 側で、ckpts/ から TF-IDF インデックスや Vectorizer、
#   （必要なら）LLM をロードして、/ask で即問い合わせ可能にする設計。
# - プロセス起動時に一度だけロード → リクエスト毎に再利用（高速化）。
# - 注意: 大規模モデルや巨大インデックスの場合は起動時間・メモリ使用量に影響。
#   必要に応じて遅延ロードやバックグラウンドタスクでのプリウォームを検討。
# ============================================================
pipe = QAPipeline()

# ============================================================
# リクエストボディのスキーマ
# ------------------------------------------------------------
# - Pydantic(BaseModel) により型検証・バリデーションが自動化。
# - /docs でのインタラクティブUIや OpenAPI にも反映される。
#   例:
#   {
#     "query": "社外からVPNに接続する手順は？",
#     "top_k": 3
#   }
# ============================================================
class AskReq(BaseModel):
    query: str        # ユーザの自然文クエリ
    top_k: int = 3    # 上位何件のコンテキストを取得するか（既定=3）

@app.get("/health", include_in_schema=False)
def health():
    # あると便利：最低限の準備確認（任意）
    ok_files = all([
        Path("ckpts/tokenizer.json").exists(),
        Path("ckpts/llm.pt").exists()
    ])
    return {"status": "ok" if ok_files else "degraded", "ckpts": ok_files}

# ============================================================
# エンドポイント: /ask
# ------------------------------------------------------------
# - 入力: AskReq（query, top_k）
# - 出力: pipe.ask(...) の戻り値（dictを想定）
#   例: {
#         "query": "...",
#         "results": [
#           {"rank": 1, "doc_id": 0, "score": 0.83, "text": "...", "source": "..."},
#           ...
#         ],
#         "answer": "..."
#       }
# - 注意:
#   - 戻り値のスキーマを固定したい場合は response_model=... を追加すると堅牢。
#   - 例外時の扱いを制御する場合は try/except で HTTPException を返す設計に拡張。
# ============================================================
@app.post("/ask")
def ask(req: AskReq):
    # QAPipeline 内部の処理（典型例）:
    # 1) ベクトル検索: req.query を TF-IDF 変換 → cos 類似で top_k を取得
    # 2) コンテキスト組み立て: ヒット文書の本文・出典をまとめる
    # 3) 生成（任意）: MiniGPT 等にプロンプトを渡し、コンテキスト制約付きで回答生成
    # 4) 辞書化して返却
    return pipe.ask(req.query, top_k=req.top_k)
