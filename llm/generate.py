# -*- coding: utf-8 -*-
"""
テキスト生成ユーティリティ（最小実装 / 学習用）

- 役割: 保存済みのミニ GPT 風モデル (`MiniGPT`) と語彙 (`WordTokenizer`) を読み込み、
  プロンプトからサンプリングで続きを生成する。
- 生成戦略: 温度付きソフトマックス + top-k 事前フィルタ + 簡易リピティションペナルティ + n-gram反復禁止。
- 前提:
    * `ckpts/tokenizer.json` に WordTokenizer を保存済み
    * `ckpts/llm.pt` に学習済みパラメータを保存済み（`state_dict`, `block_size`, `vocab_size` を含む）
- 想定用途: 実運用ではなく、「生成の基本的な流れ」を学ぶための参照実装。

主なハイパーパラメータ:
- `max_new_tokens` : 生成する最大トークン数。早期終了条件により短く終わることもある。
- `temperature`    : 温度。>0 で確率的サンプリング、0 で貪欲（argmax）。
- `repetition_penalty` : 既出トークンのロジットを割る（弱める）係数。>1 で反復を抑制。
- `top_k` : 上位 k 語以外の候補を事前に無効化（ロジットを極小へ）。
- `no_repeat_ngram` : 直近の (n-1) に候補を足した n-gram が過去に出ていたら避ける（最大 10 回までリトライ）。

処理の概観:
1) トークナイザ・モデルのロード
2) プロンプトを ID 列にエンコード
3) 反復: 直近 `block_size` をモデルに入力→次トークンのロジットを取得
4) ロジットにヒューリスティクス（反復ペナルティ, top-k, 温度）を適用しサンプル
5) EOS や改行2連など簡易終了条件で break
6) ID 列をデコードして返却
"""

import torch
from llm.model import MiniGPT
from llm.tokenizer_word import WordTokenizer


def _top_k_filter(logits: torch.Tensor, k: int = 40) -> torch.Tensor:
    """上位 k 以外のロジットを極小値に置換してサンプリング対象から外す。

    形状:
      - 入力: `logits` … `[batch, vocab]` または `[..., vocab]` を想定
      - 出力: 同形状

    実装メモ:
      - `torch.topk(logits, k)` で各デバイスごとに上位 k のしきい値（最小要素）を取り、
        それ未満を大きな負値（ここでは `-1e10`）へ置換する。
      - 温度ソフトマックス前に呼ぶことで、確率質量を上位 k に集中させる。
    """
    if k is None or k <= 0:
        # k 指定なし（= フィルタしない）場合はそのまま返す
        return logits

    # v: 上位 k 個の値（…の集合）、_ : そのインデックス（未使用）
    v, _ = torch.topk(logits, k)
    # しきい値（上位 k の中で最小の値）を最後の次元に合わせてブロードキャスト可能な形に
    thr = v[..., -1, None]

    # しきい値未満の位置を極小値に置換（= サンプリングでほぼ選ばれなくする）
    return torch.where(logits < thr, torch.full_like(logits, -1e10), logits)


@torch.no_grad()  # 生成時は勾配不要（速度とメモリ節約）
def generate_text(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.5,
    repetition_penalty: float = 1.2,
    top_k: int = 40,
    no_repeat_ngram: int = 3,
) -> str:
    """プロンプトから続きのテキストを生成して返す。

    Args:
        prompt: 生成の起点となるテキスト。
        max_new_tokens: 追加で生成する最大トークン数。
        temperature: サンプリング温度。0 だと argmax（決定的）、大きいほど多様。
        repetition_penalty: 既出トークンを不利にする係数 (>1 で抑制)。
        top_k: softmax 前に上位 k 以外を切り捨てる（高速化と品質のトレードオフ）。
        no_repeat_ngram: n-gram の完全反復を禁止する n（3 なら tri-gram を禁止）。

    Returns:
        生成後テキスト（`prompt` を含む）。
    """
    # --- 1) トークナイザ & チェックポイントの読み込み
    tok = WordTokenizer.load("ckpts/tokenizer.json")
    ckpt = torch.load("ckpts/llm.pt", map_location="cpu")

    # モデルをチェックポイントのメタ情報で初期化
    model = MiniGPT(vocab_size=ckpt["vocab_size"], block_size=ckpt["block_size"])
    model.load_state_dict(ckpt["state_dict"])  # 学習済み重みをロード
    model.eval()  # 推論モード（Dropout などを無効化）

    # プロンプトをトークン ID 配列に変換し、形状を `[batch=1, seq_len]` に
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    # EOS（文終端）トークン ID（存在しない設計の場合もあるので None 許容）
    id_eos = tok.tok2id.get("<EOS>")

    # n-gram 反復チェック関数
    def violates(ids: torch.Tensor, cand: torch.Tensor, n: int) -> bool:
        """`ids` の末尾 (n-1) に `cand` を連結した n-gram が過去に出現していたら True。

        - `ids`: `[1, cur_len]` の ID 列（バッチ 1 固定前提の簡易実装）
        - `cand`: 次候補 ID（スカラーテンソル想定）
        - `n`: n-gram 長（1 以下なら無効）

        計算量は O(cur_len * n)。小モデル向けの簡易版。
        """
        if n <= 1 or ids.size(1) < n - 1:
            return False
        seq = ids[0].tolist()
        # 末尾 (n-1) + 候補 で n-gram を構成
        pat = seq[-(n - 1) :] + [int(cand)]
        # 過去に同じ n-gram が出ていたか走査
        for i in range(len(seq) - n + 1):
            if seq[i : i + n] == pat:
                return True
        return False

    # --- 2) 逐次サンプリングでトークンを積み増す
    for _ in range(max_new_tokens):
        # 入力は直近 block_size トークン（長文はウィンドウで切り詰め）
        logits, _ = model(idx[:, -model.block_size :])  # 出力形状: [B=1, T, V]
        logits = logits[:, -1, :]  # 直近トークンの次トークン用ロジット: [1, V]

        # 2-1) repetition penalty: これまで出たトークンのスコアを弱める
        #     ※ ここでは「割る」方式。`>1` で小さくなる。引き算や温度調整など別実装もある。
        for t in set(idx[0].tolist()):
            logits[0, t] /= repetition_penalty

        # 2-2) top-k フィルタ（上位 k 以外をほぼ無効化）。確率のしっぽを刈る。
        logits = _top_k_filter(logits, top_k)

        # 2-3) サンプリング or 貪欲選択
        if temperature > 0:
            # 温度付き softmax で確率化
            probs = torch.softmax(logits / temperature, dim=-1)

            # n-gram 禁止を満たすまで最大 10 回リトライ
            tries = 0
            while True:
                next_id = torch.multinomial(probs, num_samples=1)  # 形状 [1,1]
                if not violates(idx, next_id[0, 0], no_repeat_ngram) or tries > 10:
                    break
                # 禁止 n-gram に当たった候補の確率を 0 にして再正規化
                # 注意: 次の 1 行はインデクシングの簡易記法で、`probs[0, next_id[0,0]] = 0` と同義にするのが安全。
                probs[0, next_id] = 0  # 簡易/学習用（本番は要厳密化）
                probs = probs / probs.sum()
                tries += 1
        else:
            # 決定的生成（温度 0）: もっとも確率の高いトークンを選ぶ
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # 形状 [1,1]

        # 2-4) EOS で早期終了（トークナイザ設計に依存）
        if id_eos is not None and int(next_id) == id_eos:
            break

        # 2-5) 生成列に連結
        idx = torch.cat([idx, next_id], dim=1)

        # 2-6) 簡易な整形による終了判定（改行 2 連で“段落終わり”とみなす）
        out_txt = tok.decode(idx[0].tolist())
        if out_txt.endswith("\n\n"):
            break

    # --- 3) ID -> テキストへ戻して返却
    return tok.decode(idx[0].tolist())
