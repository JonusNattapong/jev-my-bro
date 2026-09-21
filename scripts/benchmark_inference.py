from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import laya

from jevbro.schema import read_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark jev-my-bro inference latency and throughput")
    parser.add_argument("--model", default="artifacts/laya-model")
    parser.add_argument("--data", default="data/test.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--report", default="artifacts/benchmark.json")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return values[index]


def memory_snapshot(device: str) -> dict:
    result: dict[str, float | str] = {}
    try:
        import psutil

        result["process_rss_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        result["process_rss_mb"] = -1.0
    if device.startswith("cuda"):
        import torch

        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return result


def main() -> None:
    args = parse_args()
    cases = read_cases(args.data)
    samples = [(case["state"], case["questions"]) for case in cases]
    load_start = time.perf_counter()
    agent = laya.Agent(args.model, device=args.device)
    if args.device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1000

    first_start = time.perf_counter()
    agent.predict(*samples[0])
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    first_request_ms = (time.perf_counter() - first_start) * 1000
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    results = []

    for batch_size in batch_sizes:
        for index in range(args.warmup):
            state, questions = samples[index % len(samples)]
            agent.predict(state, questions)
        if args.device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        latencies: list[float] = []
        group_times: list[float] = []
        for repeat in range(args.repeats):
            start = time.perf_counter()
            for offset in range(batch_size):
                state, questions = samples[(repeat * batch_size + offset) % len(samples)]
                item_start = time.perf_counter()
                agent.predict(state, questions)
                latencies.append((time.perf_counter() - item_start) * 1000)
            if args.device.startswith("cuda"):
                import torch

                torch.cuda.synchronize()
            group_times.append(time.perf_counter() - start)

        total_requests = batch_size * args.repeats
        results.append(
            {
                "batch_size": batch_size,
                "execution_mode": "sequential Agent.predict calls grouped as a batch",
                "requests": total_requests,
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "mean_ms": statistics.mean(latencies),
                "throughput_requests_per_second": total_requests / max(1e-9, sum(group_times)),
                "group_p50_ms": percentile([value * 1000 for value in group_times], 0.50),
                "memory": memory_snapshot(args.device),
            }
        )

    report = {
        "model": args.model,
        "dataset": args.data,
        "device": args.device,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cold_start": {
            "model_load_ms": model_load_ms,
            "first_request_ms": first_request_ms,
            "memory_after_first_request": memory_snapshot(args.device),
        },
        "note": "The jev-my-bro Laya runtime exposes no tensor-batch API here; batch sizes measure grouped sequential requests.",
        "results": results,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
