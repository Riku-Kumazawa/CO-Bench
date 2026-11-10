"""Download and pin the checkpoint without loading it or running inference."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Qwen/Qwen3.8-27B"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO_ID, revision=args.revision, files_metadata=True)
    if not info.sha:
        raise RuntimeError("Could not resolve the model revision.")
    print(f"Downloading {REPO_ID}@{info.sha} to {model_dir}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=info.sha,
        local_dir=model_dir,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "README.md", "LICENSE*"],
        max_workers=4,
    )
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    expected_sizes = {f.rfilename: f.size for f in info.siblings}
    for name in shards:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != expected_sizes[name]:
            raise RuntimeError(f"Incomplete weight shard: {path}")
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        if not (model_dir / name).is_file():
            raise RuntimeError(f"Missing model file: {name}")
    manifest = {
        "repo_id": REPO_ID,
        "revision": info.sha,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "weight_bytes": sum((model_dir / name).stat().st_size for name in shards),
        "shards": {name: expected_sizes[name] for name in shards},
    }
    (model_dir / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {len(shards)} weight shards; revision {info.sha}")


if __name__ == "__main__":
    main()
