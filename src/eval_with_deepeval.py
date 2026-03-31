"""
DeepEval script to evaluate ChatML tool-calling inference data.

Evaluation targets (based on the fixed 5-turn message structure):
  messages[0] = system
  messages[1] = user
  messages[2] = assistant  → tool_calls   (Eval 1: Tool Call Accuracy)
  messages[3] = tool        → tool result
  messages[4] = assistant  → final answer (Eval 2: Final Response Alignment)

Judge model is loaded from configs/config.yaml via LiteLLMModel.

Output artifact layout:
  <output_dir>/
    run_info.json   – reproducibility metadata (timestamp, git, args, config)
    results.jsonl   – one JSON line per evaluated test case
    summary.json    – aggregate pass-rate and score statistics per eval type
"""

import json
from pathlib import Path
from typing import Union

import yaml
from config_morpher import ConfigMorpher
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import GEval
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from export_helpers import *

console = Console()

DATA_PATH = "examples/exps/exp_20260331155146/data.jsonl"
CONFIG_PATH = "configs/config.yaml"

# Keys that map directly to LiteLLMModel constructor parameters.
_LITELLM_KNOWN_KEYS = {"model", "api_key", "base_url", "temperature"}


# ─────────────────────────────────────────────────────────────────────────────
# Load judge model from config via ConfigMorpher
# ─────────────────────────────────────────────────────────────────────────────

def load_judge_model(config_path: str) -> LiteLLMModel:
    cfg = yaml.safe_load(Path(config_path).read_text())
    morpher = ConfigMorpher(cfg)
    model_kwargs: dict = morpher.fetch("model")

    # Keys not recognised by LiteLLMModel (e.g. name, max_tokens) are
    # forwarded to litellm's completion() via generation_kwargs.
    generation_kwargs = {
        k: v for k, v in model_kwargs.items()
        if k not in _LITELLM_KNOWN_KEYS and k != "name"
    }

    # Explicitly disable logprobs-related params for Gemini models
    generation_kwargs["logprobs"] = None
    generation_kwargs["top_logprobs"] = None

    return LiteLLMModel(
        model=model_kwargs["model"],
        api_key=model_kwargs.get("api_key"),
        temperature=model_kwargs.get("temperature", 0),
        generation_kwargs=generation_kwargs or None,
    )


judge = load_judge_model(CONFIG_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_records(path: str, index_expr: Union[int, str] = ":") -> list[dict]:
    if isinstance(index_expr, int):
        index_expr = str(index_expr)

    with open(path, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # Note: eval() is used for slice parsing. Mitigation applied via restricted globals, but inherently unsafe for untrusted input.
    result = eval(f"dataset[{index_expr}]", {"__builtins__": {}}, {"dataset": dataset})
    return result if isinstance(result, list) else [result]


def build_test_cases(
    record: dict,
) -> tuple[LLMTestCase, LLMTestCase]:
    messages: list[dict] = record["messages"]
    tools: list[dict] = record["tools"]

    system_msg        = messages[0]
    user_msg          = messages[1]
    assistant_tc_msg  = messages[2]
    tool_result_msg   = messages[3]
    assistant_fin_msg = messages[4]

    raw_tool_call = assistant_tc_msg["tool_calls"][0]
    tool_call_payload = {
        "name": raw_tool_call["function"]["name"],
        "arguments": json.loads(raw_tool_call["function"]["arguments"]),
    }

    tc_test_case = LLMTestCase(
        input=user_msg["content"],
        actual_output=json.dumps(tool_call_payload, ensure_ascii=False, indent=2),
        context=[
            f"System prompt:\n{system_msg['content']}",
            f"Available tools:\n{json.dumps(tools, ensure_ascii=False, indent=2)}",
        ],
    )

    align_test_case = LLMTestCase(
        input=user_msg["content"],
        actual_output=assistant_fin_msg["content"],
        context=[
            f"Tool result:\n{tool_result_msg['content']}",
        ],
    )

    return tc_test_case, align_test_case


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# (defined once and reused across all records)
# ─────────────────────────────────────────────────────────────────────────────

tool_call_metric = GEval(
    name="Tool Call Alignment",
    model=judge,
    criteria=(
        "Given the user's question and the available tool schemas, evaluate whether "
        "the assistant's tool call is correct. Specifically assess:\n"
        "1. Call decision: Was it appropriate to call a tool at all?\n"
        "2. Tool selection: Is the chosen tool name the most suitable one?\n"
        "3. Argument validity: Do the arguments conform to the tool's JSON schema "
        "(correct types, required fields present, no hallucinated fields)?\n"
        "4. Argument relevance: Do the argument values make semantic sense for the user's question?"
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    threshold=0.7,
)

alignment_metric = GEval(
    name="Post-Tool Alignment",
    model=judge,
    criteria=(
        "Given the user's question and the tool result provided as context, "
        "evaluate whether the assistant's final response meets these criteria:\n"
        "1. Faithfulness: Does the response only use information from the tool result "
        "(no hallucinated facts beyond what is provided)?\n"
        "2. Completeness: Does it fully answer the user's question based on the tool result?\n"
        "3. Relevance: Is the response focused and free of irrelevant information?\n"
        "4. Clarity: Is the answer clear and easy to understand?"
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    threshold=0.7,
)

# ─────────────────────────────────────────────────────────────────────────────
# Run evaluations over all (or a slice of) records
# Usage examples:
#   python src/eval_with_deepeval.py               # all records
#   python src/eval_with_deepeval.py --index 0     # first record only
#   python src/eval_with_deepeval.py --index 0:5   # first 5 records
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tool calling using DeepEval.")
    parser.add_argument(
        "-i",
        "--index",
        type=str,
        default=":",
        help="Index expression to select records (e.g., '0', '0:5', ':'). Defaults to ':'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="examples/evals/eval",
        help="Base path for the output artifact directory. A timestamp suffix is appended automatically.",
    )
    args = parser.parse_args()

    index_expr = args.index
    records = load_records(DATA_PATH, index_expr)

    # Prepare output directory and run_info
    output_dir = make_output_dir(args.output)
    cfg = yaml.safe_load(Path(CONFIG_PATH).read_text())
    model_kwargs: dict = ConfigMorpher(cfg).fetch("model")
    run_info_path = write_run_info(
        output_dir,
        data_path=DATA_PATH,
        config_path=CONFIG_PATH,
        index_expr=index_expr,
        model_kwargs=model_kwargs,
    )

    console.print(
        Panel(
            f"[bold]Data:[/bold]    {DATA_PATH}  "
            f"[dim](index={index_expr!r}, {len(records)} record(s))[/dim]\n"
            f"[bold]Output:[/bold]  {output_dir}",
            title="[bold cyan]DeepEval Run[/bold cyan]",
            expand=False,
        )
    )

    tc_cases, align_cases = [], []
    for record in records:
        tc, al = build_test_cases(record)
        tc_cases.append(tc)
        align_cases.append(al)

    console.print(Rule(f"[bold cyan]Eval 1 · Tool Call Accuracy[/bold cyan]  [dim]({len(tc_cases)} cases)[/dim]"))
    tc_result: EvaluationResult = evaluate(
        test_cases=tc_cases,
        metrics=[tool_call_metric],
        display_config=DisplayConfig(verbose_mode=False, show_indicator=False),
    )
    tc_key = eval_type_key(tool_call_metric.name)
    tc_results_path = write_results(output_dir, tc_result, tc_key)

    console.print(Rule(f"[bold cyan]Eval 2 · Final Response Alignment[/bold cyan]  [dim]({len(align_cases)} cases)[/dim]"))
    align_result: EvaluationResult = evaluate(
        test_cases=align_cases,
        metrics=[alignment_metric],
        display_config=DisplayConfig(verbose_mode=False, show_indicator=False),
    )
    align_key = eval_type_key(alignment_metric.name)
    align_results_path = write_results(output_dir, align_result, align_key)

    summary_path = write_summary(output_dir, [tc_key, align_key])

    console.print(Rule())
    artifact_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    artifact_table.add_column("Artifact", style="cyan")
    artifact_table.add_column("Path", style="dim")
    artifact_table.add_row("run_info.json", str(run_info_path))
    artifact_table.add_row(f"results_of_{tc_key}.jsonl", str(tc_results_path))
    artifact_table.add_row(f"results_of_{align_key}.jsonl", str(align_results_path))
    artifact_table.add_row("summary.json", str(summary_path))
    console.print(
        Panel(artifact_table, title=f"[bold green]Artifacts[/bold green]  [dim]{output_dir}[/dim]", expand=False)
    )
