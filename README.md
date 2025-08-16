# minimal-llm-rag

PyCon JP 2025 デモ用の最小 RAG 実装です。

---

## セットアップ

```bash
git clone <your-repo-url>
cd minimal-llm-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

※ `jq` は OS コマンドなので別途インストールしてください。

* macOS: `brew install jq`
* Ubuntu: `sudo apt install jq`

---

## 実行方法

### インデックス構築

```bash
python -m rag.build_index
```

### API 起動

```bash
uvicorn app.api:app --reload
```

---

## サンプル質問

```bash
# VPN 関連
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"社外からVPNに接続する手順は？","top_k":3}' | jq -r .answer

# 勤怠 関連
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"勤怠はどうやって記録しますか？","top_k":3}' | jq -r .answer

# メール 関連
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"社内メールは何を使いますか？","top_k":3}' | jq -r .answer
```

---

## 実行例

**VPN 関連**

```
■ 回答
- 社外から社内ネットワークへ接続する場合は GlobalProtect を使用します。
- ログイン時は社内アカウントとパスワードを入力し、Duo による多要素認証を行います。
- 端末にクライアントが無い場合は IT ポータルからインストールしてください。
■ 出典: 001_入館とVPN
```

**勤怠 関連**

```
■ 回答
- 勤怠は始業前に打刻し、退勤時にも打刻します。
- 休暇は上長承認が必要です。申請は人事システムから行ってください。
■ 出典: 002_勤怠と休暇
```

**メール 関連**

```
■ 回答
- 社内メールは Outlook を使用します。
- 大容量ファイルの共有は社内共有ドライブを利用してください。
- チャットは Microsoft Teams を利用します。
■ 出典: 003_メールとチャット
```