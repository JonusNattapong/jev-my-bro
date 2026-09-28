import json
from pathlib import Path

import numpy as np

from jevbro.config import load_config
from jevbro.evaluate import multiclass_nll
from jevbro.publish import main as publish_main
from jevbro.train import split_paths


def test_colab_config_is_loadable_and_reproducible() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "colab-t4.yaml")
    assert config["seed"] == 42
    assert config["device"] == "cuda"
    assert config["train"].endswith("train.jsonl")


def test_toolcall_config_trains_on_both_corpora_and_never_on_test_splits() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "colab-th1200-toolcall.yaml")
    train_paths = split_paths(config["train"])
    validation_paths = split_paths(config["validation"])

    assert "data/th_curated_1200/train.jsonl" in train_paths
    assert "data/tool_call_80/train.jsonl" in train_paths
    assert config["score_cumulative_weight"] > 0
    for path in train_paths + validation_paths:
        assert not path.endswith("test.jsonl"), f"locked test split must stay untouched: {path}"
        assert not path.endswith("calibration.jsonl"), f"calibration split must stay untouched: {path}"


def test_split_paths_dedupes_and_rejects_empty() -> None:
    assert split_paths("a.jsonl, b.jsonl ,a.jsonl") == ["a.jsonl", "b.jsonl"]
    try:
        split_paths(" , ")
    except ValueError:
        return
    raise AssertionError("empty spec must raise")


def test_th500_config_keeps_ordinal_score_objective_enabled() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "colab-th500.yaml")
    assert config["score_cumulative_weight"] > 0
    assert 0 < config["score_class_balance_beta"] < 1


def test_th560_config_selects_ordinal_balanced_dataset() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "colab-th560.yaml")
    assert config["train"].endswith("data/th_curated_560/train.jsonl")
    assert config["score_cumulative_weight"] == 1.0


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
