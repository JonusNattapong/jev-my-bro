import json
from pathlib import Path

import numpy as np

from jevbro.config import load_config
from jevbro.evaluate import multiclass_nll
from jevbro.publish import main as publish_main
from jevbro.train import split_paths, summarize_head_diagnostics


def test_head_diagnostics_flag_a_head_that_only_predicts_the_mean() -> None:
    """A head that ignores its input must show zero spread and zero separation.

    This is the measured failure: the shipped th1200 risk head returns the training mean for
    every request, which no accuracy or QWK figure exposes.
    """
    flat_scores = {level: [2.4, 2.4, 2.4] for level in range(5)}
    flat_noul = {qid: {"true": [0.63, 0.63], "false": [0.63, 0.63]} for qid in ("needs_review", "prohibited")}

    score_report, noul_report = summarize_head_diagnostics(flat_scores, flat_noul)

    assert score_report["spread_max_minus_min"] == 0.0
    assert noul_report["prohibited"]["separation"] == 0.0
    assert noul_report["needs_review"]["separation"] == 0.0


def test_head_diagnostics_reward_a_head_that_conditions_on_input() -> None:
    scores = {0: [0.4, 0.5], 1: [1.0], 2: [2.0], 3: [3.0], 4: [3.8, 4.0]}
    noul = {
        "needs_review": {"true": [0.9, 0.8], "false": [0.2, 0.3]},
        "prohibited": {"true": [0.7], "false": [0.2, 0.1, 0.2]},
    }

    score_report, noul_report = summarize_head_diagnostics(scores, noul)

    assert score_report["spread_max_minus_min"] > 1.0
    assert score_report["by_gold_level"]["0"] < score_report["by_gold_level"]["4"]
    assert noul_report["needs_review"]["separation"] > 0.3
    assert noul_report["prohibited"]["separation"] > 0.3


def test_head_diagnostics_tolerate_a_missing_class() -> None:
    score_report, noul_report = summarize_head_diagnostics(
        {}, {"prohibited": {"true": [], "false": [0.2]}}
    )

    assert score_report["spread_max_minus_min"] is None
    assert noul_report["prohibited"]["separation"] is None
    assert noul_report["prohibited"]["gold_false_mean"] == 0.2


def test_enforcement_gate_criteria_name_the_metrics_the_training_log_reports() -> None:
    """The gate must reference the exact metric names validation_metrics emits."""
    gate = (Path(__file__).parents[1] / "docs" / "ENFORCEMENT_GATE.md").read_text(encoding="utf-8")
    for metric in (
        "validation.score.spread_max_minus_min",
        "validation.noul.prohibited.separation",
        "validation.noul.needs_review.separation",
    ):
        assert metric in gate, f"gate criterion must name {metric}"
    assert "1.474" in gate, "gate must report the corrected risk spread"



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
