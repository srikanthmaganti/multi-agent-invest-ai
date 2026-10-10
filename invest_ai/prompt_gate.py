"""CI gate comparing a candidate prompt with a pinned baseline on holdout data."""

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from invest_ai.prompt_evaluate import (
    DATASET_DIR,
    _start_generator,
    evaluate_advisor,
    evaluate_router,
    load_cases,
    validate_prompt_uris,
)
from invest_ai.prompts import ADVISOR_PROMPT_NAME, ROUTER_PROMPT_NAME


GATED_METRICS = {
    "router": {
        "accuracy": 1.0,
    },
    "advisor": {
        "citation_coverage_rate": 1.0,
        "no_unknown_citations_rate": 1.0,
        "risky_phrase_checks_pass_rate": 1.0,
        "groundedness_score": 0.8,
        "answer_relevancy_score": 0.8,
        "context_precision": 0.75,
        "context_recall": 0.75,
    },
}


def compare_metrics(
    baseline: dict[str, float],
    candidate: dict[str, float],
    minimums: dict[str, float],
) -> dict[str, Any]:
    """Require minimum scores and no regression from baseline."""
    checks = {}
    for metric, minimum in minimums.items():
        if metric not in baseline or metric not in candidate:
            raise ValueError(f"Missing required benchmark metric: {metric}")
        checks[metric] = {
            "baseline": baseline[metric],
            "candidate": candidate[metric],
            "minimum": minimum,
            "passes_minimum": candidate[metric] >= minimum,
            "no_regression": candidate[metric] >= baseline[metric],
        }
    return {
        "passed": all(
            check["passes_minimum"] and check["no_regression"]
            for check in checks.values()
        ),
        "checks": checks,
    }


def _run_task(
    task: str,
    baseline_uri: str,
    candidate_uri: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    generator = _start_generator()
    evaluator = evaluate_router if task == "router" else evaluate_advisor
    baseline_metrics, _, baseline_run_id = evaluator(
        baseline_uri, cases, "holdout", generator
    )
    candidate_metrics, _, candidate_run_id = evaluator(
        candidate_uri, cases, "holdout", generator
    )
    comparison = compare_metrics(
        baseline_metrics,
        candidate_metrics,
        GATED_METRICS[task],
    )
    return {
        "task": task,
        "baseline_uri": baseline_uri,
        "candidate_uri": candidate_uri,
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        **comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("router", "advisor"), required=True)
    parser.add_argument("--baseline-uri", required=True)
    parser.add_argument("--candidate-uri", required=True)
    args = parser.parse_args()
    if args.baseline_uri == args.candidate_uri:
        parser.error("Baseline and candidate prompt URIs must be different.")

    load_dotenv()
    import mlflow

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    prompt_name = ROUTER_PROMPT_NAME if args.task == "router" else ADVISOR_PROMPT_NAME
    for label, uri in (
        ("baseline", args.baseline_uri),
        ("candidate", args.candidate_uri),
    ):
        if not uri.startswith(f"prompts:/{prompt_name}/"):
            parser.error(
                f"{label} URI must refer to the {prompt_name!r} prompt, got {uri!r}."
            )

    try:
        validate_prompt_uris([args.baseline_uri, args.candidate_uri])
        cases = load_cases(DATASET_DIR / f"{args.task}_holdout.jsonl")
        result = _run_task(
            args.task,
            args.baseline_uri,
            args.candidate_uri,
            cases,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Prompt benchmark gate could not run: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(json.dumps(result, indent=2))
    if not result["passed"]:
        print("Prompt benchmark gate FAILED.", file=sys.stderr)
        raise SystemExit(1)
    print("Prompt benchmark gate PASSED.")


if __name__ == "__main__":
    main()
