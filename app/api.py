# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.pipeline import QAPipeline

app = FastAPI(title="minimal-llm-rag")

# 引数なしで自己完結ロードできるようにした
pipe = QAPipeline()

class AskReq(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ask")
def ask(req: AskReq):
    return pipe.ask(req.query, top_k=req.top_k)
