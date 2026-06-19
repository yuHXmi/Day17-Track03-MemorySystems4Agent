from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    import json
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected or not answer:
        return 0.0
    ans_lower = answer.lower()
    matches = 0
    for item in expected:
        if item.lower() in ans_lower:
            matches += 1
    if matches == len(expected):
        return 1.0
    elif matches == 0:
        return 0.0
    else:
        return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    if not answer:
        return 0.0
    r_score = recall_points(answer, expected)
    if len(answer) > 400:
        return max(0.0, r_score - 0.1)
    return r_score


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """
    import os
    total_tokens_only = 0
    total_prompt_tokens = 0
    recall_scores = []
    quality_scores = []
    total_compactions = 0

    # Clean profile files in states/profiles if Advanced Agent
    if hasattr(agent, "profile_store"):
        if agent.profile_store.root_dir.is_dir():
            for p in agent.profile_store.root_dir.glob("*.md"):
                try:
                    p.unlink()
                except Exception:
                    pass

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]
        recall_questions = conv["recall_questions"]

        # Feed turns
        for turn in turns:
            agent.reply(user_id, conv_id, turn)

        # Retrieve usage and compaction statistics
        total_tokens_only += agent.token_usage(conv_id)
        total_prompt_tokens += agent.prompt_token_usage(conv_id)
        total_compactions += agent.compaction_count(conv_id)

        # Ask recall questions in a fresh thread (cross-session recall)
        recall_thread_id = f"{conv_id}-recall"
        for rq in recall_questions:
            question = rq["question"]
            expected = rq["expected_contains"]

            reply_dict = agent.reply(user_id, recall_thread_id, question)
            ans = reply_dict.get("content", "")

            r_pts = recall_points(ans, expected)
            q_pts = heuristic_quality(ans, expected)

            recall_scores.append(r_pts)
            quality_scores.append(q_pts)

    # Compute average scores
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Memory growth
    mem_growth = 0
    if hasattr(agent, "profile_store"):
        if agent.profile_store.root_dir.is_dir():
            for p in agent.profile_store.root_dir.glob("*.md"):
                mem_growth += p.stat().st_size

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_tokens_only,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=mem_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    from tabulate import tabulate
    headers = [
        "Agent Name", "Agent Tokens Only", "Prompt Tokens Processed",
        "Cross-Session Recall", "Response Quality", "Memory Growth (bytes)", "Compactions"
    ]
    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2%}",
            f"{r.response_quality:.2%}",
            r.memory_growth_bytes,
            r.compactions
        ])
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """
    import copy
    config = load_config(Path(__file__).resolve().parent.parent)

    # Load datasets
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    print(f"Loaded {len(standard_convs)} standard conversations.")
    print(f"Loaded {len(stress_convs)} stress conversations.")

    # Determine offline vs live
    import os
    force_offline = True
    if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        if os.getenv("FORCE_OFFLINE", "false").lower() != "true":
            force_offline = False

    print(f"Running benchmark in {'OFFLINE' if force_offline else 'LIVE'} mode...")

    # 1. Standard Benchmark
    print("\n=== RUNNING STANDARD BENCHMARK ===")
    baseline_std = BaselineAgent(config, force_offline=force_offline)
    advanced_std = AdvancedAgent(config, force_offline=force_offline)

    row_baseline_std = run_agent_benchmark("Baseline Agent", baseline_std, standard_convs, config)
    row_advanced_std = run_agent_benchmark("Advanced Agent", advanced_std, standard_convs, config)

    print("\nStandard Benchmark Results:")
    print(format_rows([row_baseline_std, row_advanced_std]))

    # 2. Long-Context Stress Benchmark
    print("\n=== RUNNING LONG-CONTEXT STRESS BENCHMARK ===")
    # Create configuration with lower threshold to ensure compaction is active
    stress_config = copy.copy(config)
    stress_config.compact_threshold_tokens = 300
    stress_config.compact_keep_messages = 2

    baseline_stress = BaselineAgent(stress_config, force_offline=force_offline)
    advanced_stress = AdvancedAgent(stress_config, force_offline=force_offline)

    row_baseline_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, stress_config)
    row_advanced_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_convs, stress_config)

    print("\nLong-Context Stress Benchmark Results:")
    print(format_rows([row_baseline_stress, row_advanced_stress]))


if __name__ == "__main__":
    main()
