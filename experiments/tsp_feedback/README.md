# TSPフィードバック実験

初期解固定・3条件×9反復の実験定義を用意しています。モデル生成は未実行です。

- 編集中の設定：`drafts/next/experiment.yaml`
- 介入文：`drafts/next/prompts/interventions/`
- 確定した設定：`versions/v001/`
- 実行待ちのrun：`runs/ready_v001/`
- 操作手順：リポジトリの `RUN_EXPERIMENTS.md`

```bash
uv run --frozen python experiment.py check experiments/tsp_feedback/runs/ready_v001
uv run --frozen python experiment.py run experiments/tsp_feedback/runs/ready_v001
```

Qwenサーバーは別ターミナルで `./scripts/serve_qwen.sh` を起動します。
プロンプトや設定を変更して実行するときは新しいversionとrunを作成してください。
