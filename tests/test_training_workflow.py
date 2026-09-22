import json
from pathlib import Path

import numpy as np

from jevbro.config import load_config
from jevbro.evaluate import multiclass_nll
from jevbro.publish import main as publish_main


def test_colab_config_is_loadable_and_reproducible() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "colab-t4.yaml")
    assert config["seed"] == 42
    assert config["device"] == "cuda"
    assert config["train"].endswith("train.jsonl")


def test_nll_is_finite_for_normalized_soft_target() -> None:
    value = multiclass_nll(np.array([0.75, 0.25]), np.array([0.5, 0.5]))
    assert np.isfinite(value)
    assert value > 0


def test_publish_dry_run_validates_checkpoint(tmp_path: Path, capsys) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"test")
    (tmp_path / "rl_agent_config.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "tokenizer").mkdir()
    publish_main(["--checkpoint", str(tmp_path), "--repo-id", "owner/model", "--dry-run"])
    assert "validated checkpoint" in capsys.readouterr().out
