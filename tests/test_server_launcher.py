"""Check the launcher environment without starting vLLM or using GPUs."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/serve_qwen.sh"


class ServerLauncherTests(unittest.TestCase):
    def make_environment(self, root, with_ninja=True):
        bin_dir = root / "venv/bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python"
        python.write_text('#!/bin/sh\nprintf "ninja=%s\\n" "$(command -v ninja)"\nprintf "args=%s\\n" "$*"\nninja --version\n')
        python.chmod(0o755)
        if with_ninja:
            ninja = bin_dir / "ninja"
            ninja.write_text('#!/bin/sh\nprintf "fake-ninja-version\\n"\n')
            ninja.chmod(0o755)
        model = root / "model"
        model.mkdir()
        (model / "checkpoint.json").write_text("{}")
        # Isolate PATH so a globally installed ninja cannot hide the regression.
        system_bin = root / "system-bin"
        system_bin.mkdir()
        (system_bin / "dirname").symlink_to("/usr/bin/dirname")
        env = {
            **os.environ,
            "PATH": str(system_bin),
            "COBENCH_VLLM_PYTHON": str(python),
            "COBENCH_MODEL_DIR": str(model),
            "CUDA_VISIBLE_DEVICES": "2,3,4,5",
            "COBENCH_TP_SIZE": "4",
            "COBENCH_MAX_MODEL_LEN": "65536",
        }
        return env, bin_dir

    def test_virtualenv_ninja_is_visible_to_server_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, bin_dir = self.make_environment(Path(tmp))
            result = subprocess.run(["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"ninja={bin_dir / 'ninja'}", result.stdout)
            self.assertIn("fake-ninja-version", result.stdout)
            self.assertIn("--max-model-len 65536", result.stdout)
            self.assertNotIn("--enforce-eager", result.stdout)

    def test_missing_ninja_is_reported_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, _ = self.make_environment(Path(tmp), with_ninja=False)
            result = subprocess.run(["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ninja not found", result.stderr)
            self.assertIn("uv pip install --python", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
