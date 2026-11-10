# 実験定義・プロンプト・実行履歴の管理

`experiment.py` は、初期解・親の選び方・介入文・反復数を設定で切り替える実験ランナーです。
以前の `tsp_continue.py` と保存済みの結果はそのまま利用できます。

## 保存構成

```text
experiments/tsp_feedback/
├── README.md
├── drafts/next/                 # 編集する場所
│   ├── experiment.yaml
│   ├── README.md                # 仮説・選定理由など
│   └── prompts/
│       ├── refine.txt
│       └── interventions/{support,neutral,switch}.txt
├── initials/qwen_tsp_01/        # 登録済み初期解と来歴。IDは上書きしない
├── versions/v001/               # 確定した定義。上書き不可
│   ├── version.json            # 元バージョン、理由、作成日時、ハッシュ
│   ├── experiment.yaml         # 既定値まで展開した設定
│   ├── source.yaml             # 確定時に入力された設定
│   ├── prompts/                # プロンプト全文
│   └── initials/               # その時点の初期解・来歴のコピー
└── runs/<run_id>/
    ├── snapshot/               # 使用したversionの完全なコピー
    ├── manifest.json           # 進捗、親、環境、実装と成果物のハッシュ
    ├── plan.json               # 全候補の条件、反復、step、seed
    ├── task.json               # データのハッシュとdev対象インデックス
    ├── evaluation.json
    ├── initials/<initial_id>/  # このrunの初期コードとdev評価
    ├── branches/<initial_id>/<policy>/<condition>/repeat_001/step_001/
    │   ├── prompt.txt          # 実際に送信した全文
    │   ├── request.json
    │   ├── response.json
    │   ├── response.txt
    │   ├── reasoning.txt       # 応答に含まれている場合
    │   ├── solution.py         # コードを抽出できた場合
    │   ├── dev.json
    │   ├── dev.rows.json       # 評価チェックポイント
    │   └── dev.log
    └── reports/
        ├── scores.csv
        ├── comparisons.csv
        ├── usage.csv
        ├── summary.json
        └── figures/            # PNG / SVG / PDF
```

編集対象はdraftです。設定を残したい時点で必ず `freeze` してください。
確定versionは既存名への保存を拒否し、ファイルを手で編集した場合もハッシュ検証で検出します。
同じversionから何回でも別runを作成できます。過去runの再開はsnapshotを使うため、draftや別versionの編集に影響されません。
データ本体はコピーせず、場所・全ファイルのハッシュ・dev選択を保存し、実行時に検証します。データを更新する場合は旧データも別ディレクトリに保持してください。
versionsは初期解も含め自己完結するので、共有initialsの後日の編集にも依存しません。
この機能はGitのcommitを自動作成しません。draft編集中の一文字単位の履歴はGit等で管理し、比較したい定義はfreezeで保存します。

## 新しい実験を作る

```bash
cd /home/rocklee/CO-Bench
uv run --frozen python experiment.py init experiments/my_feedback \
  --source runs/tsp_qwen_latest_10_submissions/initial \
  --task 'Travelling salesman problem' --initial-id qwen_tsp_01
```

生成・評価は実行しません。初期解と、確認可能な評価来歴を取り込みます。
今回のTSP初期解は旧継続実験のマニフェストでコード・結果・現在のデータハッシュを照合し、保存済み評価を利用できます。
単独の `solution.py` も `--source` に指定できます。その場合、初期dev評価はrun時に実行します。
署名のない旧 `feedback.json` は評価条件を確認できないため、そのままキャッシュとして扱いません。

## 親の選び方と生成数

| `parent_policies` | 親 |
|---|---|
| `[initial]` | 毎回、同じ初期解 |
| `[latest]` | 同じ条件・反復の直前の候補。悪化・失敗も引き継ぐ |
| `[best]` | 初期解を含む、その経路で最高devのコード。同点は既存の最高解を維持 |
| `[initial, latest, best]` | 各方式を独立した経路としてすべて実行 |

`best` はコード不備・コード抽出失敗の候補を親に選びません。選ばれた親のコードとdevフィードバックを渡し、直前の別候補や別経路の情報は追加しません。
`latest` で構文エラーならそのコードとエラー、コードを抽出できなければ直前の最終応答と失敗理由を渡します。以前のreasoningは渡しません。

```yaml
schema_version: 1
name: my_feedback
task: Travelling salesman problem
data_dir: /home/rocklee/CO-Bench/data
initials:
  - id: qwen_tsp_01
    directory: ../../initials/qwen_tsp_01  # drafts/nextからの相対パス
parent_policies: [initial]
repeats: 9
steps: 1
prompts:
  template: prompts/refine.txt
  conditions:
    support: prompts/interventions/support.txt
    neutral: prompts/interventions/neutral.txt
    switch: prompts/interventions/switch.txt
seeds:
  base: 101
  paired_across_conditions: true
generation:
  model: Qwen/Qwen3.8-27B
  api_base: http://127.0.0.1:8000/v1
  thinking: true
  reasoning_effort: medium
  max_tokens: 32768
  temperature: 1.0
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  presence_penalty: 0.0
  repetition_penalty: 1.0
  timeout_seconds: 7200
  server_wait_seconds: 600
evaluation:
  split: dev
  unsplit_policy: all_as_dev
  timeout_seconds: 10
  cpu_num: 1
  feedback_length: 64
```

`steps` は初期解を除く新規生成数、`repeats` は独立した反復経路数です。
新規生成総数は **初期解数 × 親方式数 × 条件数 × repeats × steps** です。

| 実験 | 親方式 | repeats | steps | 3条件・初期解1つの新規生成数 |
|---|---|---:|---:|---:|
| 初期解固定の独立生成 | initial | 9 | 1 | 27 |
| 今回と同じ長さの連続改善 | latest | 1 | 9 | 27 |
| 最高解を親にして連続改善 | best | 1 | 9 | 27 |
| 連続改善を3経路ずつ | latest | 3 | 9 | 81 |

seedは `base + (repeat - 1) * steps + step - 1` で固定します。
`paired_across_conditions: true` では同じ反復・stepの各条件に同じseedを割り当てます。
falseでは条件ごとに `条件の並び順 × repeats × steps` を加えます。初期解と親方式の間では共通seedを使います。
同じseedでもAPIやGPU実装による完全な再現や、条件間で同じ乱数の影響になることは保証されません。
実行順は初期解→親方式→反復→step→条件の順です。各stepで条件を順番に回します。

## プロンプトを編集し、確定する

`drafts/next/prompts/interventions/*.txt` を直接編集します。neutralを空ファイルにすると介入文なしです。
条件の追加・名称変更は `experiment.yaml` の `prompts.conditions` でも指定してください。
共通テンプレートには以下の5変数を使います。変数の欠落・誤記は生成前に検出します。

```text
{problem_description}
{previous_code}
{dev_feedback}
{intervention}
{timeout}
```

テンプレート自体にPython等の波括弧を書く場合は `{{` と `}}` にエスケープします。
介入文とコードは置換後に再解釈されないため、通常の波括弧を使えます。

```bash
uv run --frozen python experiment.py plan experiments/my_feedback/drafts/next \
  --preview-dir /tmp/my_feedback_preview

uv run --frozen python experiment.py freeze experiments/my_feedback/drafts/next \
  --version v001 --reason '初期解固定、3条件を各9回生成する比較'

uv run --frozen python experiment.py prepare experiments/my_feedback/versions/v001 \
  --run-id first_run
```

`plan` は生成総数、初期評価の再利用可否、介入文、最初の候補とseedを表示します。
`--preview-dir` を付けると、最初の初期解を使った条件別の送信文を出力します。初期評価が未実施ならプレースホルダーを表示します。
この3コマンドはモデル呼び出しもソルバー評価も行いません。

## 実行と再開

Qwenサーバーは別ターミナルで従来のコマンドを使って起動します。

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 ./scripts/serve_qwen.sh
```

```bash
uv run --frozen python experiment.py check experiments/my_feedback/runs/first_run
uv run --frozen python experiment.py run experiments/my_feedback/runs/first_run

# 中断後
uv run --frozen python experiment.py resume experiments/my_feedback/runs/first_run
```

`run` と `resume` は同じ再開動作をします。保存済みの応答を再生成せず、評価はケース単位のチェックポイントから再開します。
送信済みの可能性があるのに応答が保存されていない候補は停止し、自動再送しません。
構文エラー・コード抽出失敗・出力上限はその候補を失敗として0点にし、専用の修正生成は追加しません。
保存済みの親・seed・送信文・応答・コード・評価結果の改変を検出します。
実行コード・Python環境・データを変更した場合は、既存runの継続を拒否します。
初期評価はコード・データ・dev対象・タイムアウト・評価実装・Python環境等が一致する場合に再利用し、表示するfeedbackは現在の表示長で再作成します。

評価対象はdevのみです。dev/test分割がある問題ではdevに選ばれた入力だけを実行し、参照値との正規化は元の入力インデックスを維持します。
TSPのように分割がなければ `all_as_dev` で従来の全入力を使います。`unsplit_policy: error` とすると、分割のない問題を誤って全件評価することを防げます。
元のCO-Bench採点関数、ファイルごとの平均→ファイル間平均、タイムアウト+1秒の猶予を維持します。

## 設定を変更して別バージョンを作る

```bash
uv run --frozen python experiment.py fork experiments/my_feedback/versions/v001 --draft prompt_variant

# drafts/prompt_variant/experiment.yaml と prompts/ を編集

uv run --frozen python experiment.py diff \
  experiments/my_feedback/versions/v001 experiments/my_feedback/drafts/prompt_variant

uv run --frozen python experiment.py freeze experiments/my_feedback/drafts/prompt_variant \
  --version v002 --reason 'supportの介入文を変更'
```

`fork` は元バージョンを記録します。v002はv001の全文を保持したまま別定義として保存され、両方から実行できます。
同じdraftを繰り返しfreezeする場合も、既存versionと異なる名前が必要です。直前に確定したversionを自動で派生元に記録します。派生元は `--parent v001` で明示することもできます。

```bash
uv run --frozen python experiment.py list experiments/my_feedback
uv run --frozen python experiment.py report experiments/my_feedback/runs/first_run
```

`report` はモデル・ソルバーを呼び出さず集計と図を再作成します。
`initial` の図は条件別のスコア分布、`latest/best` の図は経路ごとのスコア推移です。最高値で置き換えず各提出のスコアを表示します。
CSVには実際の親、seed、初期解・親からの差、失敗数、時間を記録します。summaryには経路ごとの最新・最終・観測最高の候補を分けて保存します。
連続改善のstepや同一コードのケース数を独立した実験反復として扱わないでください。

## 別の初期解・問題

```bash
uv run --frozen python experiment.py import-initial experiments/my_feedback qwen_tsp_02 \
  --source /path/to/another/solution.py --task 'Travelling salesman problem'
```

draftの `initials` に登録したIDとディレクトリを追加します。
forkしたdraft内の初期解を別IDに差し替える場合、共有initialsへの相対パスは `../../initials/<id>` です。
初期解コードを変更する際は新しいIDで登録し、確定versionの中は編集しません。
問題を変える場合は別の実験フォルダを `init --task ...` で作成し、その問題の初期解を指定する方法を推奨します。

## オフライン検証

```bash
uv run --frozen python -m unittest discover -s tests -v
```

生成は模擬し、小さな問題だけ実際の評価ワーカーで実行します。Qwenへの生成リクエストは送りません。
