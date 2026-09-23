"""Run JevBench's published tasks against a local Jev/Laya checkpoint.

The JevBench repository is intentionally kept outside this repository. This
runner imports its canonical task loader, result type, serial runner, ledger,
and summarizer so the scoring path stays theirs rather than becoming a second
local benchmark implementation.

Only the published JevBench tasks can be run locally. The official board also
uses held-out tasks that are not distributed in the JevBench repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", default=None, help="comma-separated JSONL files; defaults to all published tasks")
    parser.add_argument("--results", type=Path, default=Path("private/jevbench-public/results.jsonl"))
    parser.add_argument("--raw-dir", type=Path, default=Path("private/jevbench-public/raw"))
    parser.add_argument("--ledger", type=Path, default=Path("private/jevbench-public/ledger.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("private/jevbench-public/manifest.json"))
    parser.add_argument("--public-export", type=Path, default=Path("private/jevbench-public/summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-s", type=float, default=0.0)
    return parser.parse_args()


def import_jevbench(root: Path):
    root = root.resolve()
    if not (root / "jevbench").is_dir():
        raise SystemExit(f"--jevbench-dir does not contain jevbench/: {root}")
    sys.path.insert(0, str(root))
    from jevbench.adapters.base import DecisionResult, build_question
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.summarize import public_export, summarize
    from jevbench.tasks import dataset_hash, load_jsonl

    return DecisionResult, build_question, Ledger, Runner, public_export, summarize, dataset_hash, load_jsonl


def build_adapter(DecisionResult, build_question, model_path: Path, device: str):
    class JevMyBroAdapter:
        name = "jev_my_bro_laya"
        cost_basis = "local_gpu_no_provider_tariff"
        price_input_per_m = None
        price_output_per_m = None

        def __init__(self):
            self.model_path = str(model_path)
            self.device = device
            self._agent = None
            self.laya_version = None

        def load(self):
            if self._agent is None:
                import laya

                self._agent = laya.Agent(self.model_path, device=self.device)
                self.laya_version = getattr(laya, "__version__", None)
            return self._agent

        def run(self, task):
            result = DecisionResult(
                adapter=self.name,
                ok=False,
                probs_source="native",
                model=self.model_path,
            )
            request = {
                "state": task.state,
                "questions": {"decision": build_question(task)},
            }
            result.request_body = request
            try:
                agent = self.load()
            except Exception as exc:  # noqa: BLE001 - recorded as a failed item
                result.error = f"load failed: {type(exc).__name__}: {str(exc)[:250]}"
                return result

            started = time.perf_counter()
            try:
                output = agent.predict(request["state"], request["questions"])
            except Exception as exc:  # noqa: BLE001 - recorded as a failed item
                result.latency_s = time.perf_counter() - started
                result.error = f"{type(exc).__name__}: {str(exc)[:300]}"
                return result

            result.latency_s = time.perf_counter() - started
            result.raw = {
                "response": output,
                "runtime": {
                    "device": self.device,
                    "laya": self.laya_version,
                    "probability_origin": "native-softmax",
                },
            }
            result.usage = dict((output or {}).get("usage") or {})
            answer = ((output or {}).get("answers") or {}).get("decision")
            try:
                if not isinstance(answer, dict) or answer.get("type") != task.question["type"]:
                    raise ValueError("missing or mistyped answers.decision")
                if task.question["type"] == "noul":
                    probability = float(answer["noul"])
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError(f"noul out of range: {probability}")
                    result.probs = {"yes": probability, "no": 1.0 - probability}
                else:
                    probabilities = answer["probabilities"]
                    if not isinstance(probabilities, dict):
                        raise ValueError("missing probabilities")
                    result.probs = {str(key): float(value) for key, value in probabilities.items()}
            except (KeyError, TypeError, ValueError) as exc:
                result.error = f"answer parse failed: {exc}"
                return result
            result.ok = True
            return result

        def reserve_estimate(self, task):
            return 0.0

    return JevMyBroAdapter()


def main() -> int:
    args = parse_args()
    (
        DecisionResult,
        build_question,
        Ledger,
        Runner,
        public_export,
        summarize,
        dataset_hash,
        load_jsonl,
    ) = import_jevbench(args.jevbench_dir)

    if args.tasks:
        task_paths = [Path(item.strip()) for item in args.tasks.split(",") if item.strip()]
    else:
        task_paths = [
            args.jevbench_dir / "datasets/public/easy.jsonl",
            args.jevbench_dir / "datasets/public/original.jsonl",
            args.jevbench_dir / "datasets/public/hard.jsonl",
        ]
    tasks = [task for path in task_paths for task in load_jsonl(str(path))]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no JevBench tasks found")

    adapter = build_adapter(DecisionResult, build_question, args.model, args.device)
    adapter.load()
    ledger = Ledger(str(args.ledger), cap_usd=0.0)
    runner = Runner(adapter, ledger, raw_dir=str(args.raw_dir), default_reserve_usd=0.0)
    print(
        f"[jevbench] model={args.model} device={args.device} "
        f"published_tasks={len(tasks)} dataset_hash={dataset_hash(tasks)}",
        flush=True,
    )
    records = runner.run_all(tasks, results_path=str(args.results), delay_s=args.delay_s)
    failed = sum(record["status"] == "failed" for record in records)
    print(f"[jevbench] done: {len(records)}/{len(tasks)} attempted, {failed} failed", flush=True)

    summary = summarize(tasks, records, ledger.charged, headline_only=False)
    args.public_export.parent.mkdir(parents=True, exist_ok=True)
    args.public_export.write_text(
        json.dumps(public_export(summary, tasks, records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "benchmark": "JevBench published tasks",
                "model": str(args.model),
                "device": args.device,
                "task_files": [str(path) for path in task_paths],
                "dataset_hash": dataset_hash(tasks),
                "n_planned": len(tasks),
                "n_attempted": len(records),
                "n_failed": failed,
                "cost_basis": adapter.cost_basis,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if len(records) == len(tasks) else 3


if __name__ == "__main__":
    raise SystemExit(main())
