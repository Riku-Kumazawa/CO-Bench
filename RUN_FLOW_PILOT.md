# Flow shop: 共通初期解＋3条件の予備実験

この実験は GPT-6.1 Sol による **初期解1回＋改善3回、合計4生成**です。
4候補は初期解、方針支持 (`support`)、追加指示なし (`neutral`)、方針転換 (`switch`)。
初期解のコードとdev結果を共有し、改善時には追加文だけを変えます。
既存の `main.py` / `RUN_LOCAL.md` によるQwen実行とは別のエントリーポイントです。

## 固定した設定

| 項目 | 設定 |
| --- | --- |
| 問題 | Flow shop scheduling、既存の12サイズすべて |
| 分割 | `config.py` の既存分割、dev 60件 / test 60件 |
| モデル | `gpt-6.1-sol`、Responses API |
| 推論 | `medium`、sampling設定やAPIのseedは指定しない |
| サービス | Standard (`service_tier=default`) |
| 出力上限 | 32,768トークン、推論を含む |
| 入力上限 | APIによるカウントで20,000トークン。超過時は送信せず停止 |
| 費用枠 | $2、税・ローカル計算費用を除く |
| キャッシュ | explicitモード、breakpointなし。キャッシュ読み書きを無効化 |
| API再試行 | なし (`max_retries=0`)、1生成リクエストの待ち時間上限30分 |
| 評価 | 1ケース10秒のwall-clock上限、候補の起動とimportも含む |
| CPU | 評価を直列実行。ワーカーと子プロセスを利用可能な1コアに固定 |
| スレッド | BLAS等の環境変数を1に固定。プロンプトでも1スレッドを要求 |
| ライブラリ | 標準ライブラリ、NumPy、SciPy、OR-Toolsをプロンプトで指定 |
| 評価の乱数 | ベースseed=0からケースごとに固定、Python/NumPyに設定 |
| エラー修正 | なし。候補を複数生成して選ぶ処理もなし |
| 集約 | ケースのエラーは0、ファイル内平均→12ファイルの平均 |
| 採用候補 | 初期解と改善案をdevスコアで比較。同点なら初期解 |

プロンプトは `prompts/flow_initial.txt`、`prompts/flow_refine.txt` に保存しています。
介入文は `flow_experiment.py` の `INTERVENTIONS` にあります。
前回コードとdevのスコア・エラーを渡し、前回応答の説明や内部推論は渡しません。
それぞれ独立したuserメッセージ1つで、`previous_response_id` 等による履歴共有はありません。

全サイズの評価時間の上限目安は、4候補×(60 dev＋60 test)×10秒＝80分＋処理の付随時間。
モデルの生成時間は別です。各ケースは完了次第次に進むので、毎回10秒かかるとは限りません。
既存の汎用Evaluatorは全件をまとめて実行するため、今回は同じ `load_data` / `eval_func`
を呼ぶ専用ワーカーを追加し、devだけを先に実行します。既存Evaluatorの1秒の猶予は使いません。

## 1. 環境の準備とオフライン確認

```bash
cd /home/rocklee/CO-Bench
uv sync --frozen
uv run --frozen python flow_experiment.py check
```

`check` はAPIにアクセスせず、データ分割・依存関係を確認して
`runs/flow_preflight/` に4条件のプロンプトプレビューと `check.json` を保存します。
改善プレビュー内の初期コード・dev結果はプレースホルダーです。
実験時には各候補フォルダの `prompt.txt` に実際の送信文が保存されます。

任意で、LLMなしの評価動作確認もできます。出力先は未作成のディレクトリを指定します。

```bash
uv run --frozen python flow_experiment.py smoke --output-dir runs/flow_smoke_manual
```

固定の恒等順列を全120件で評価します。これはモデルの性能測定ではありません。

## 2. APIキーを同じ端末で入力

OpenAI PlatformでAPI課金とモデルへのアクセスが有効なプロジェクトのキーを使用します。
キーはソース・チャット・コマンド引数には記載しません。

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
```

入力は画面に表示されず、キーそのものはコマンド履歴に入りません。
設定はその端末の子プロセスにだけ有効です。

生成せずに認証・モデル参照・初期入力のトークンカウントを確認します。

```bash
uv run --frozen python flow_experiment.py check --online
```

オンライン確認が通っても、実際の生成時の残高・クォータ・レート制限による失敗はありえます。
APIエラー本文にはキーが含まれる場合があるため、実験ログには例外の種類とHTTPステータスだけを保存します。

## 3. 実験を開始（ここから有料生成）

```bash
uv run --frozen python flow_experiment.py run --output-dir runs/flow_sol_pilot_001
```

出力先は未作成のディレクトリを指定します。実行中は候補名・処理段階が表示されます。
評価のケース別進捗は各候補の `dev.log` / `test.log` に追記されます。
生成順序は `initial` → `support` → `neutral` → `switch`。
最初に初期解のdevを評価し、そこから3条件を独立に生成・dev評価します。
全4候補の生成後に、それぞれをtestで評価します。test情報は一切プロンプトに入りません。

費用は入力$2 / 出力$10（100万トークンあたり、2026-11-08確認）で計算します。
各生成前にAPIの入力トークン数を確認し、その入力＋最大出力の料金が残予算以内か確認します。
固定上限では4回すべて最大まで使っても生成料金の見積もりは **$1.47072**。
usageの推論トークンはoutput_tokensの内数なので二重加算しません。
費用ログは料金表に基づく見積もりです。実際の請求はOpenAI側の利用明細で確認してください。
料金が改定された場合は新しい実験を始める前に設定を更新します。

## 4. 結果を確認

```text
runs/flow_sol_pilot_001/
  run.json                 設定、コード・データのハッシュ、生成の状態、usage、費用
  summary.json             候補別dev/testスコア、成功率、devで選んだ候補
  scores.csv               候補別のスコア・成功率・費用・時間
  size_scores.csv          サイズ群ごとのスコア
  scores.png / scores.pdf  提出1→2の図
  initial/                 以下はsupport/neutral/switchにも保存
    prompt.txt             実際の送信文
    request.json           キーを含まないAPIリクエスト
    response.json          APIの生応答とusage
    response.txt           応答の可視テキスト
    solution.py            抽出したコード（抽出失敗時はなし）
    dev.json / test.json   ケースごとの結果と集約
    dev.rows.json          ケース単位の再開用チェックポイント
    test.rows.json
    dev_feedback.txt       LLM向けdev評価情報
    dev.log / test.log     評価の進捗
```

図の左列は提出候補そのもの、右列はdevで採用した候補です。
devの右列はbest-so-farで単調非減少になります。
testの右列も **devの採用判断に従う** ため、testスコアは低下する場合があります。
testの最高値を選ぶ処理はありません。両条件が同スコアなら線が重なります。
1初期解・各条件1改善の予備実験なので、誤差帯や統計的有意性は表示しません。

図とCSVだけを再生成できます（API呼び出し・再評価なし）。

```bash
uv run --frozen python flow_experiment.py report runs/flow_sol_pilot_001
```

## 5. 失敗と中断・再開

- 初期解のコード抽出/構文失敗、出力途中終了、dev全件失敗：記録して停止（終了コード2）。
- 初期解がdevの一部だけ失敗：エラーを全条件に同じように渡して続行。
- 改善案の抽出/構文失敗、出力途中終了：その条件のスコアを0として記録し、他条件を続行。
- ケース単位の例外や時間切れ：そのケースを0として集約。提出を除外しない。
- API障害：コード性能の0点と混同せず実験を停止。自動再試行なし。
- 実行基盤の障害：候補の0点にせず停止。

```bash
uv run --frozen python flow_experiment.py resume runs/flow_sol_pilot_001
```

保存済み応答は再生成しません。完了した評価とケース単位の結果も再利用します。
コード・プロンプト・データ・依存関係・設定が変わった場合は再開を拒否します。
同じ出力先で複数の実行プロセスを起動することもできません。
送信直前に状態を保存するため、**送信した可能性があるのに応答が保存されていない候補は再送しません**。
通信断や生成中の中断ではこの状態になります。自動で再開できない場合はログとAPI利用明細を確認し、
別の実験としてやり直すか判断してください。状態ファイルを削除して再送しないでください。
usageが保存されなかったリクエストの費用は、予約した最大費用として扱います。

終了後、端末のキーを解除できます。

```bash
unset OPENAI_API_KEY
```

## 実装と検証上の範囲

評価ワーカーは新しいPythonプロセスを限定した環境変数で起動し、APIキーやAPIクライアントを渡しません。
各候補のコードはケースごとに独立した子プロセスで実行します。
ただし、この評価器は既存コードと同様にOS全体を隔離するコンテナではありません。
利用ライブラリの限定はプロンプト上の制約であり、importを強制的に禁止する仕組みではありません。
1コアへの固定は行いますが、候補が独自にスレッド数や乱数を設定する場合があります。
wall-clockの揺れもあるため、僅差の解釈には後続の再評価が必要です。

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_flow_experiment.py' -v
```

実際のSDKのHTTP通信を模擬するテストで、実験分岐・予算・再開・エラー処理を検証します。
テストとオフラインsmokeは課金される生成を行いません。

公式資料：
[モデル](https://developers.openai.com/api/docs/models/gpt-6.1-sol)、
[料金](https://developers.openai.com/api/docs/pricing)、
[入力トークン計数](https://developers.openai.com/api/docs/guides/token-counting)、
[キャッシュ設定](https://developers.openai.com/api/docs/guides/prompt-caching)。
