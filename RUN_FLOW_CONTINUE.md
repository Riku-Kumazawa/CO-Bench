# Flow shop：各フィードバックで初期解込み10提出まで継続

既存の `runs/flow_sol_pilot_001` の初期解と3改善案を引き継ぎます。
**各条件10提出＝初期解1つ＋改善9回。共通初期解を共有するので合計28生成、追加は24生成です。**
元の4生成の実験フォルダは更新せず、コピーした別フォルダで続行します。

## 確定した設定

- モデル、プロンプト、推論強度medium、入出力トークン上限、ライブラリ、評価分割・時間制限・CPUは前回と同じ。
- 3条件の追加文を、各改善生成で毎回付ける。neutralは追加文なし。
- 各条件内でdev最高スコアのコードとそのdev評価結果を次の改善元にする（Greedy Refinement）。
- devが同点または悪化なら以前の最高解を保持。testは採用・親選択に使わない。
- 改善元の候補IDを各生成の `run.json` レコードの `parent` に記録。
- 他条件の候補や結果は共有しない。共通初期解だけを共有する。
- 提出順は各ラウンドでsupport→neutral→switch。各条件の提出3から10を追加。
- 専用のエラー修正・候補の追加生成・API自動再試行はなし。失敗候補も提出回数を消費する。
- 失敗案が保持されない場合、次回LLMには保持した最高解とその評価だけを渡す。失敗案のコード・エラーを別途追加することはしない。
- 新しい24生成とdev評価を全て終えてから、新しい24候補をtest評価する。
- 既存候補のdev/test評価はコピーした結果を使い、再実行しない。
- 既存test結果は既に観察済みだが、それを親選択や改善プロンプトに使わない。今回の継続は同じ初期解を使った探索的な追加実験。

最初の継続元は次のとおりです。

| 条件 | 提出3の改善元 | 理由 |
|---|---|---|
| support | support | 提出2でdevが初期解より改善 |
| neutral | neutral | 提出2でdevが初期解より改善 |
| switch | initial | 提出2はタイムアウト込みのdevが低いため、初期解を保持 |

## 準備済みフォルダを確認（API呼び出しなし）

```bash
cd /home/rocklee/CO-Bench
uv run --frozen python flow_continue.py check runs/flow_sol_10_submissions
```

これはコード・プロンプト・データ・環境・コピーした結果のハッシュを検証します。
28生成の計画、各条件の完了提出数、親選択、費用を表示し、CSVと図を更新します。
API呼び出しと候補の再評価は行いません。

新しく継続用フォルダを作る場合は以下を使います。既に準備済みのフォルダには再実行しません。

```bash
uv run --frozen python flow_continue.py prepare \
  --source runs/flow_sol_pilot_001 \
  --output-dir runs/flow_sol_10_submissions \
  --submissions 10 \
  --budget-usd 10
```

`prepare`もAPIを呼びません。元実験は完了した4生成実験である必要があります。
実装・プロンプト・評価方法が前回から変わっている場合は、黙って引き継がず停止します。

## APIキーを設定し、開始（有料生成）

前回と同じ端末でキーが有効なら再入力は不要です。

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'

uv run --frozen python flow_continue.py run runs/flow_sol_10_submissions
```

実行中は候補名・全体の生成数・改善元を表示します。例えば `support_03` は支持条件の提出3。
各候補のケース別評価進捗は `support_03/dev.log` 等に保存します。
既存の `flow_experiment.py resume` は4生成実験用なので、今回の継続には使いません。

## 費用と所要時間

OpenAI APIのStandard料金を2026-11-09に再確認し、入力$2・出力$10／100万トークンを使用。
キャッシュは引き続き無効。推論トークンは出力の内数として集計します。

- 前回4生成の実績：$0.226576。
- 前回4生成の平均単価で24生成を見積もると、追加約$1.36、総額約$1.59。
- 継続時にはコードの長さや推論量が変わるため、実測費用は変動します。
- 追加24回が全て入力20,000・出力32,768トークンなら、追加$8.82432、総額$9.050896。
- **既存分込みの予算枠は$10**。各送信前に、残予算がその入力＋最大出力をカバーできるか確認します。
- 認証失敗・通信断等で応答が未保存の場合、最大料金を予約した扱いにし、自動再送しません。

各候補はdev 60件＋test 60件です。追加24候補は2,880ケース。
全ケースが10秒使う場合、評価だけで約8時間＋処理の付随時間、モデル生成時間は別です。
前回は多くのケースで制限時間近くまで探索したため、数時間の実行を想定してください。
実行するならtmux等の長時間維持できる端末が適しています。

## 中断・再開・結果

```bash
uv run --frozen python flow_continue.py resume runs/flow_sol_10_submissions
```

保存済み応答・完了した評価・ケースごとの評価チェックポイントを再利用します。
送信した可能性があるのに応答が保存されていない生成は再送しません。
コード、データ、プロンプト、依存環境、設定、元結果のコピーが変わると停止します。
障害の種類を `run.json` の `last_error` に保存し、APIエラーの生本文やキーは保存しません。

出力先には以下を保存します。

- `run.json`：28候補の計画、元実験の情報、生成状態、各候補の親、費用。
- `source_run.json`：元の実験マニフェストのコピー。
- `summary.json`：各条件の完了提出数、最終保持解、候補別スコア、探索経路。
- `scores.csv`：条件・提出回数・親・候補スコア・保持解スコア・正常終了率。
- `size_scores.csv`：サイズ別の候補スコア。
- `costs.csv`：生成候補ごとの費用・usage・時間。共通初期解の二重計上を防ぐため、費用は別表。
- `scores.png` / `scores.pdf`：横軸1〜10の提出回数。左列は提出候補、右列はdevで保持した解。
- `initial/`, `support/`, `neutral/`, `switch/`：コピーした元の4候補。
- `support_03/`〜`switch_10/`：追加した24候補のプロンプト・応答・コード・dev/test結果。

testの保持解スコアはdevでの採用判断に従うため、単調非減少とは限りません。
途中で新候補のtestが未評価なら、その部分は空白として表示します。

図とCSVだけを再作成する場合：

```bash
uv run --frozen python flow_continue.py report runs/flow_sol_10_submissions
```

オフラインの継続機能テスト：

```bash
uv run --frozen python -m unittest discover -s tests -p 'test_flow_continue.py' -v
```

公式料金：[OpenAI API pricing](https://developers.openai.com/api/docs/pricing)。
