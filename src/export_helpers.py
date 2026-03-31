import json
from datetime import datetime
from pathlib import Path

from deepeval.evaluate.types import EvaluationResult

from utils.git_info import get_git_info

# ─────────────────────────────────────────────────────────────────────────────
# Artifact export helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_output_dir(base: str) -> Path:
    """Create a timestamped output directory derived from *base*."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    raw = Path(base)
    output_dir = raw.parent / f"{raw.name}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_run_info(
    output_dir: Path,
    *,
    data_path: str,
    config_path: str,
    index_expr: str,
    model_kwargs: dict,
) -> Path:
    """Write run_info.json following the same structure as the raft project."""
    redacted_config = {k: v for k, v in model_kwargs.items() if k != "api_key"}

    run_info = {
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(),
        "args": {
            "data_path": data_path,
            "config": config_path,
            "output": str(output_dir),
            "index": index_expr,
        },
        "config": {
            "model": redacted_config,
        },
    }

    info_path = output_dir / "run_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    return info_path


def eval_type_key(metric_name: str) -> str:
    """Convert a metric display name to a snake_case key, e.g. 'Tool Call Accuracy' → 'tool_call_accuracy'."""
    return metric_name.lower().replace(" ", "_")


def write_results(
    output_dir: Path,
    eval_result: EvaluationResult,
    eval_type: str,
) -> Path:
    """Write per-test-case results to <eval_type>.jsonl (one JSON line per case)."""
    results_path = output_dir / f"results_of_{eval_type}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for i, tr in enumerate(eval_result.test_results):
            metrics = []
            for md in (tr.metrics_data or []):
                metrics.append({
                    "name": md.name,
                    "score": md.score,
                    "threshold": md.threshold,
                    "success": md.success,
                    "reason": md.reason,
                    "evaluation_model": md.evaluation_model,
                    "evaluation_cost": md.evaluation_cost,
                    "error": md.error,
                })

            entry = {
                "index": i,
                "success": tr.success,
                "input": tr.input,
                "actual_output": tr.actual_output,
                "context": tr.context,
                "metrics": metrics,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return results_path


def write_summary(output_dir: Path, eval_types: list[str]) -> Path:
    """Read per-eval-type jsonl files and write an aggregate summary.json."""
    buckets: dict[str, list] = {}
    for et in eval_types:
        jsonl_path = output_dir / f"results_of_{et}.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            buckets[et] = [json.loads(line) for line in f if line.strip()]

    evals_summary: dict[str, dict] = {}
    for et, entries in buckets.items():
        scores = [
            m["score"]
            for e in entries
            for m in e["metrics"]
            if m["score"] is not None
        ]
        passed = sum(1 for e in entries if e["success"])
        total = len(entries)
        evals_summary[et] = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "min_score": round(min(scores), 4) if scores else None,
            "max_score": round(max(scores), 4) if scores else None,
        }

    summary = {
        "total_cases": sum(v["total"] for v in evals_summary.values()),
        "evals": evals_summary,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path