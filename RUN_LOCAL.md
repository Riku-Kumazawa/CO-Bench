# Qwen3.8-27Bで1タスクを評価する

作業場所は `/home/rocklee/CO-Bench` です。モデル配信と評価には別々のターミナルを使います。
初期設定は、Qwen3.8-27BのBF16重み、GPU 0〜3の4枚、コンテキスト64K、thinking有効・mediumです。

## 準備の確認

```bash
cd /home/rocklee/CO-Bench
uv run --frozen python main.py --check
```

このコマンドはローカルの依存関係・チェックポイント・問題データを確認します。
サーバー起動、モデル推論、ソルバー評価は行いません。

別環境にコピーした場合は `uv sync --frozen` でCO-Bench環境を復元します。
モデルを取得し直す場合は以下を使います。ダウンロード済みファイルは再利用されます。

```bash
uv run --frozen python scripts/prepare_qwen.py
```

モデルは `models/Qwen3.8-27B/` に配置し、`checkpoint.json` にリビジョンと重みのサイズを記録します。
推論には既存の `/home/rocklee/qwen-vllm/.venv/bin/python` を使います。
起動スクリプトは推論用仮想環境の`bin`を`PATH`に追加し、初期化時に使う`ninja`を見つけられるようにします。
`ninja`が見つからない場合は、モデルをロードする前に導入コマンドを表示して停止します。

## ターミナル1：モデル配信

```bash
cd /home/rocklee/CO-Bench
./scripts/serve_qwen.sh
```

初回はモデルロードとカーネルの準備に時間がかかります。GPUメモリが使われていても、APIの起動は完了していない場合があります。
サーバーの起動完了後に、ターミナル2の評価コマンドを実行します。
`main.py`はAPIの準備完了を最大600秒待つため、初期化中に起動しても接続確認を続けます。
起動コマンドだけ表示するには `./scripts/serve_qwen.sh --dry-run` を使います。

APIは `http://127.0.0.1:8000/v1`、モデル名は `Qwen/Qwen3.8-27B` です。
評価終了後、ターミナル1で `Ctrl+C` を押してサーバーを停止するとGPUメモリが解放されます。
使用するGPUを変更する場合は、起動時に指定します。

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/serve_qwen.sh
```

4枚の使用を前提とします。VRAMは各24GBで、重みに加えて実行用メモリが必要です。
起動スクリプトはコンテキスト64K、同時生成1件で開始します。`--enforce-eager`は外しており、CUDA Graph・コンパイル最適化はvLLMの既定値を使います。
コンテキストには入力文と生成出力の両方が含まれます。出力上限を32Kにする場合、従来の32Kサーバーは再起動が必要です。

## ターミナル2：1候補を生成して評価

```bash
cd /home/rocklee/CO-Bench
uv run --frozen python main.py
```

実行内容は固定の1候補です。

1. APIにモデル名を照会し、接続先を確認する。
2. GreedyRefineの初回プロンプトで、ソルバーコードを1回生成する。
3. Pythonコードブロックと構文、`solve`関数の有無を確認する。
4. 同じコードをAircraft landing全49インスタンスで評価する。
5. dev/testスコアとエラーを保存して終了する。

改善ループ、自動再生成、生成APIのリトライは行いません。
起動待機中は`GET /v1/models`で接続を確認します。この確認ではコード生成を依頼しません。
生成上限は32,768トークン（推論過程を含む）、生成待ち時間は最大7,200秒（2時間）です。
実測7〜8.5トークン/秒では、32,768トークンの生成に約64〜78分かかる計算です。
これは時間の見積もりであり、このトークン数までに最終回答が完了することを保証するものではありません。
ソルバーの制限時間は各インスタンス10秒、CPU並列数は1です。
候補コードには標準ライブラリ、NumPy、SciPy、OR-Toolsを使うよう指示します。

設定変更の例：

```bash
uv run --frozen python main.py --reasoning-effort low --max-tokens 8192 --timeout 10
```

最初に生成から評価までの動作を確認する場合は、思考過程を無効にできます。
この設定もAPIリクエストは1回で、改善ループはありません。thinking有効時とは生成条件が異なります。

```bash
uv run --frozen python main.py --no-thinking --max-tokens 8192
```

thinking有効時はtemperature=1.0・top_p=0.95・presence_penalty=0.0、無効時は0.7・0.8・1.5を使います。
`--reasoning-effort`は`--no-thinking`時には適用しません。
思考過程を無効にしても、出力上限や時間制限に達する可能性はあります。

`--generation-timeout` はLLM生成の待ち時間、`--timeout` は各入力でのソルバーの制限時間です。
`--server-wait-timeout` は生成前のサーバー起動待ち時間です。例えば`--server-wait-timeout 1200`で最大20分待ちます。0にすると接続できない場合にすぐ終了します。
各入力の実行時間に加えて、コード生成・プロセス起動・採点の時間がかかります。

## 別タスク：Travelling salesman problem（巡回セールスマン問題）

vLLMが起動済みの場合、次のコマンドで実行できます。

```bash
cd /home/rocklee/CO-Bench
./scripts/run_tsp.sh
```

このスクリプトは前回のAircraft landingと同じ条件を明示して`main.py`を呼び出します。
Qwen/Qwen3.8-27B、thinking有効・medium、生成上限32,768トークン、生成待ち時間7,200秒、
各入力10秒、CPU並列数1、seed=0です。コード生成は1回で、改善・再生成ループはありません。
接続先は既定の`http://127.0.0.1:8000/v1`です（`COBENCH_API_BASE`または`--api-base`で変更可能）。

問題は、平面上の全都市を1回ずつ訪れて出発点に戻る経路の総距離を最小にするものです。
`solve(**kwargs)`は都市の座標`nodes`から、全都市のインデックスを重複なく並べた`{'tour': [...]}`を返します。
同じ生成コードを次の全272入力で評価します。

| データファイル | 都市数／入力 | 入力数 |
|---|---:|---:|
| `tsp500_test_concorde.txt` | 500 | 128 |
| `tsp1000_test_concorde.txt` | 1,000 | 128 |
| `tsp10000_test_concorde.txt` | 10,000 | 16 |

採点は経路の妥当性と総距離を確認し、設定ファイルの基準距離を生成経路の距離で割って正規化します。
スコアは大きいほど良く、各ファイル内で平均を取り、その3ファイルの平均が全体スコアになります。
このタスクにはdev/test分割がないため、`dev_score`と`test_score`はどちらも同じ全272件の結果です。
独立したテスト集合に対するスコアではありません。
全件が10秒近くかかる場合、評価だけで約45分にプロセス起動などの時間が加わります。コード生成時間は別途かかります。

生成・評価を開始せずに準備だけを確認する場合：

```bash
./scripts/run_tsp.sh --check
```

結果は他の実行と同様に`runs/日時-ID/`へ保存され、`run.json`の`task`で区別できます。
今回の保存コードを再評価する場合にも、タスク名の指定が必要です。

```bash
./scripts/run_tsp.sh --code runs/日時-ID/solution.py
```

## 保存物と失敗時の動作

毎回、新しい `runs/日時-ID/` を作成します。

| ファイル | 内容 |
|---|---|
| `run.json` | 設定、処理状態、生成リクエスト数、スコア、依存バージョン |
| `request.json` / `prompt.txt` | 生成リクエストとプロンプト。APIキーは保存しない |
| `response.json` / `response.txt` | モデルの応答原文 |
| `reasoning.txt` | 応答に含まれる場合の推論過程 |
| `solution.py` | 抽出したコード。構文エラーがあっても保存する |
| `feedback.json` / `feedback.txt` | 評価結果とdev/testフィードバック |
| `error.txt` | 処理が停止した場合の例外詳細 |

通信失敗・コードブロック欠落・生成上限到達・構文エラーでは、再生成せず失敗を記録します。
非ストリーミングで応答を受け取るため、通信タイムアウト時には途中までの生成テキストは保存されません。
制約違反・実行例外・タイムアウトは評価結果に含めます。
終了コードは、全件成功が0、評価に失敗インスタンスが含まれる場合は2、それ以外の失敗は1です。

保存したコードだけを再評価する場合は、モデルを起動せずに実行できます。

```bash
uv run --frozen python main.py --code runs/日時-ID/solution.py
```

接続先は `COBENCH_API_BASE` または `--api-base`、モデル名は `COBENCH_MODEL` または `--model` で変更できます。
認証が必要な接続先では `COBENCH_API_KEY` を環境変数に設定します。
サーバーのポートは `COBENCH_PORT`、推論用Pythonは `COBENCH_VLLM_PYTHON`、モデル配置先は `COBENCH_MODEL_DIR` で変更できます。
サーバーのコンテキスト上限は `COBENCH_MAX_MODEL_LEN` で変更できます。既存サーバーへ接続する際、出力上限がサーバーのコンテキスト上限以上なら生成前に停止します。
`run.json`の`local_checkpoint_revision`はローカル重みのリビジョンです。
独自のAPIへ接続した場合、配信モデルのリビジョンと一致することを保証する値ではありません。

## 準備段階で行う検証

```bash
uv run --frozen python -m unittest discover -s tests -v
bash -n scripts/serve_qwen.sh
```

テストではモデル応答と評価器をモックに置き換えます。実際のモデル推論・タスク評価は行いません。
