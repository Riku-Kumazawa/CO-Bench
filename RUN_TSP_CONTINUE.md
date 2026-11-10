# Qwen TSP：常に最新解から3条件・各10提出

既存の `runs/20261106-230111-b22819f8/solution.py` と保存済みdev評価を共通初期解として使います。
初期解のdevscoreは **0.8852062501473457**。初期解は再生成・再評価しません。
**各条件は初期解込み10提出。追加は支持9＋追加指示なし9＋方針転換9＝27生成です。**
初期解を含む独立したコードは28候補で、条件別の表では初期解を各条件の提出1として表示します。

## 固定した条件

- Qwen/Qwen3.8-27B、BF16、GPU 4枚、コンテキスト65,536、同時生成1件。
- `scripts/serve_qwen.sh` から `--enforce-eager` を削除。CUDA Graph/コンパイル最適化はvLLMの既定値を使う。
- thinking有効、reasoning effort=medium、最大出力32,768、生成待ち時間7,200秒。
- temperature=1.0、top_p=0.95、top_k=20、min_p=0、presence_penalty=0、repetition_penalty=1、seed=0。
- 標準ライブラリ、NumPy、SciPy、OR-Tools。ソルバーのCPU並列数1。
- 各入力10秒。元のCO-Bench実装の1秒猶予も維持し、元のインスタンス実行関数・採点関数を使う。
- TSPにはdev/test分割がないため、**前回と同じ全272入力をdevとして評価**。新しい分割は作らない。
  - 500都市128件、1,000都市128件、10,000都市16件。
  - 正規化スコアはファイル内平均を取り、その3ファイルの平均。失敗は0点。
- testの評価・集計・LLMへの提示は行わない。
- 改善プロンプトは既存の `prompts/flow_refine.txt` を利用。サイズ情報などの追加は行わない。
- 各改善には、**その条件の直前の提出**のコードとdevフィードバックを渡す。
  フィードバックのケース別スコア表示は元の評価器と同じく各ファイル先頭64件まで、平均は全272件。
- **devが悪化・同点・0点でも、直前の提出を次の親にする。最高解への復帰はしない。**
- 構文エラーなら直前のコードとエラーを渡す。コードを抽出できなかった場合や出力上限に達した場合は、
  直前の可視の最終応答と失敗理由を渡す。最終応答が空なら、その事実と失敗理由を渡す。
  過去の有効コードへの差し替えや、過去のreasoningの追加は行わない。
- ラウンドごとにsupport→neutral→switch。各条件の追加文を毎回付ける。neutralは追加文なし。
- コード不備・出力上限到達はその候補を0点として提出数を消費。専用の修正生成はしない。
- APIの自動再試行はなし。送信済みの可能性があるのに応答がない場合は停止し、自動再送しない。

| 条件 | 追加文 |
|---|---|
| support | I suggest continuing to refine the current algorithmic approach. |
| neutral | なし |
| switch | I suggest exploring a different algorithmic approach. |

初期コードと初期評価は以前のeager実行時のものをそのまま引き継ぎます。
今回の生成サーバー設定変更はマニフェストに記録します。
親は提出2では`initial`、提出3以降では必ず同じ条件の1つ前の提出です。

主な比較対象は各提出のdevscore推移と提出10のdevscoreです。
観測した最高devscoreも補助指標として記録しますが、親選択・最終解の選択・プロンプトには使いません。
提出10が失敗した場合も、それを最終提出として記録し、以前の最高解を最終解として選び直しません。

## 準備の確認（モデル呼び出しなし）

最新解方式の準備済みフォルダは `runs/tsp_qwen_latest_10_submissions` です。
以前の最高解保持方式の準備フォルダ `runs/tsp_qwen_10_submissions` は変更せず残してあります。
以下のコマンドは新しい最新解方式のフォルダを既定で使用します。

```bash
cd /home/rocklee/CO-Bench
./scripts/run_tsp_continue.sh check
```

コード、プロンプト、データ、環境、初期解コピーのハッシュと設定を検証します。
状態・完了提出数・最新解・補助指標の最高解を表示し、CSVを更新します。API接続や候補評価は行いません。

## ターミナル1：Qwenサーバー

```bash
cd /home/rocklee/CO-Bench
CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/serve_qwen.sh
```

GPU 2〜5の4枚を使う例です。実行前に `nvidia-smi` で使用状況を確認してください。
別の4枚を使う場合は `CUDA_VISIBLE_DEVICES` を変更します。
起動時はロードに加えてコンパイル・CUDA Graphの準備が入ります。
起動成功後の実測速度は今回の準備段階では未計測です。

## ターミナル2：追加27生成とdev評価

```bash
cd /home/rocklee/CO-Bench
./scripts/run_tsp_continue.sh
```

APIは既存と同じ `http://127.0.0.1:8000/v1` を使用します。
初回接続時は最大600秒待ちます。サーバーがまだ準備中で待ち時間を超えた場合は、
サーバー起動後に同じコマンドを実行してください。この接続確認では生成を行いません。

生成・評価中は候補名と改善元を表示します。例えば `support_02` は支持条件の提出2です。
各候補の272件の評価進捗は、その候補の `dev.log` に記録します。
生成は既存と同じ非ストリーミングで、応答完了後にreasoningを保存します。

前回のTSPは生成約17分、評価約41分でした。追加27候補は数十時間規模を見込みます。
生成速度はCUDA Graphによって変わる可能性があり、候補ごとの評価時間も変動します。
長時間維持できる端末やtmuxを利用してください。

## 中断・再開

```bash
cd /home/rocklee/CO-Bench
./scripts/run_tsp_continue.sh resume
```

`run`と`resume`はどちらも保存状態を引き継ぎます。
保存済み応答を再生成せず、完了したdev評価とケース単位の評価チェックポイントを再利用します。
送信した可能性があるのに `response.json` がない候補は自動再送しません。
コード・プロンプト・設定・データ・環境・保存した初期解が変わると再開を拒否します。
同じ出力先への多重起動も拒否します。

## 保存物

- `run.json`：27候補の計画、親、生成・評価状態、設定、ハッシュ、usage、生成時間。
- `summary.json` / `scores.csv`：各提出のdevscore、直前提出からの差、初期解からの差。
  - `latest_by_condition` / `latest_dev_scores`：条件ごとの最新提出とスコア。
  - `final_by_condition`：提出10が完了した条件の最終提出。未完了ならnull。
  - `best_observed_by_dev` / `best_observed_dev_score`：観測した最高スコアの補助指標。親には使わない。
- `usage.csv`：初期解と新候補のusage。初期解は1回だけ計上。
- `initial/`：既存初期解・応答・reasoningとdev評価のコピー。
- `support_02/`〜`switch_10/`：各追加候補の送信文、応答、reasoning、コード、dev評価。
- 候補内の `dev.rows.json`：元の目的値・エラーとケースごとの実行時間。再開に使用。

CSVだけを再作成する場合：

```bash
./scripts/run_tsp_continue.sh report
```

別の継続フォルダを用意する場合（生成・評価なし）：

```bash
uv run --frozen python tsp_continue.py prepare \
  --source runs/20261106-230111-b22819f8 \
  --output-dir runs/tsp_qwen_latest_10_submissions_other
```

通常は準備済みのフォルダを使い、`prepare`を再実行する必要はありません。

## オフラインテスト

```bash
uv run --frozen python -m unittest discover -s tests -v
```

テストではモデル呼び出しを模擬します。実際の27生成や全272件の評価は開始しません。
