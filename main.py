"""
main.py
=======

Main entry point that ties the entire CloudSync suggested-response pipeline
together:

  1. Load + merge the three source datasets, split into knowledge base / test
     set (data_loader.py).
  2. Instantiate the configured LLM provider (llm_provider.py).
  3. Build the RAG generator over the knowledge base and generate suggested
     replies for every record in the test set (generator.py).
  4. Run the LLM-as-a-Judge evaluator over the generated replies
     (evaluator.py).
  5. Write a per-response report (results.csv + results.json) and print an
     overall system average score.

Run with (on Google Colab or any environment with the dependencies
installed and API keys set as environment variables / a .env file):

    python main.py

Configuration is done entirely through environment variables (see the
CONFIG section below and the accompanying .env.example / README).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads a local .env file if present; no-op otherwise
except ImportError:
    pass

from data_loader import load_dataset_bundle
from evaluator import LLMJudge
from generator import RAGGenerator
from llm_provider import get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration -- override any of these via environment variables.
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    # Paths to the three raw dataset files (edit these to match your repo).
    "dataset_paths": [
        os.environ.get("COPILOT_DATASET_PATH", "datasets/copilot_dataset.json"),
        os.environ.get("GEMINI_DATASET_PATH", "datasets/gemini_dataset.json"),
        os.environ.get("CLAUDE_DATASET_PATH", "datasets/claude_dataset.json"),
    ],
    "dataset_source_names": ["copilot", "gemini", "claude"],

    # Train/test split.
    "test_fraction": float(os.environ.get("TEST_FRACTION", "0.3")),
    "random_seed": int(os.environ.get("RANDOM_SEED", "42")),

    # RAG settings.
    "embedding_model_name": os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
    "top_k_retrieval": int(os.environ.get("TOP_K_RETRIEVAL", "3")),
    "generation_temperature": float(os.environ.get("GENERATION_TEMPERATURE", "0.4")),
    "generation_max_tokens": int(os.environ.get("GENERATION_MAX_TOKENS", "1500")),

    # LLM provider used for BOTH generation and judging. You can also point
    # these at two different providers by instantiating get_llm_provider()
    # twice with different config dicts if you want, e.g., a cheaper model
    # to generate and a stronger model to judge.
    "llm_provider_kind": os.environ.get("LLM_PROVIDER", "standard"),  # "standard" or "huggingface"

    # Output paths.
    "output_csv_path": os.environ.get("OUTPUT_CSV_PATH", "results.csv"),
    "output_json_path": os.environ.get("OUTPUT_JSON_PATH", "results.json"),
}


def _serialize_for_csv(value: Any) -> Any:
    """Lists/dicts don't play nicely with a flat CSV, so JSON-encode them."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def build_summary(judged_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall system-level averages from the per-response scores."""
    summary = {
        "num_responses_evaluated": len(judged_df),
        "avg_intent_fulfillment_score": round(judged_df["intent_fulfillment_score"].mean(), 3),
        "avg_constraint_adherence_score": round(judged_df["constraint_adherence_score"].mean(), 3),
        "avg_overall_score": round(judged_df["overall_score"].mean(), 3),
    }
    if "intent" in judged_df.columns:
        per_intent = (
            judged_df.groupby("intent")["overall_score"]
            .mean()
            .round(3)
            .to_dict()
        )
        summary["avg_overall_score_by_intent"] = per_intent
    return summary


def run_pipeline(config: Dict[str, Any]) -> pd.DataFrame:
    """Execute the full pipeline end-to-end and return the judged DataFrame."""

    # 1. Load + merge + split the datasets.
    logger.info("Loading and merging datasets...")
    dataset_paths = config["dataset_paths"]
    missing_paths = [p for p in dataset_paths if not Path(p).exists()]
    if missing_paths:
        logger.error(
            "The following dataset files were not found: %s\n"
            "Update CONFIG['dataset_paths'] in main.py (or the corresponding "
            "*_DATASET_PATH environment variables) to point at your actual files.",
            missing_paths,
        )
        sys.exit(1)

    bundle = load_dataset_bundle(
        file_paths=dataset_paths,
        source_names=config["dataset_source_names"],
        test_fraction=config["test_fraction"],
        random_seed=config["random_seed"],
    )
    logger.info("\n%s", bundle.summary())

    # 2. Instantiate the LLM provider (Strategy Pattern).
    logger.info("Initializing LLM provider (%s)...", config["llm_provider_kind"])
    llm_provider = get_llm_provider({"provider": config["llm_provider_kind"]})

    # 3. Build the RAG generator and generate replies for the test set.
    logger.info("Building RAG generator over the knowledge base...")
    rag_generator = RAGGenerator(
        knowledge_base_df=bundle.knowledge_base_df,
        llm_provider=llm_provider,
        embedding_model_name=config["embedding_model_name"],
        top_k=config["top_k_retrieval"],
    )

    logger.info("Generating suggested replies for the test set...")
    # Limit test set to exactly 10 emails per user request
    bundle.test_df = bundle.test_df.head(10)
    generated_df = rag_generator.generate_replies_for_test_set(
        test_df=bundle.test_df,
        top_k=config["top_k_retrieval"],
        temperature=config["generation_temperature"],
        max_tokens=config["generation_max_tokens"],
    )

    # 4. Run the LLM-as-a-Judge evaluator.
    logger.info("Evaluating generated replies with LLM-as-a-Judge...")
    judge = LLMJudge(llm_provider=llm_provider)
    judged_df = judge.evaluate_dataframe(generated_df)

    return judged_df


def save_reports(judged_df: pd.DataFrame, summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Persist per-response results (CSV) and a full report (JSON)."""
    csv_df = judged_df.copy()
    
    # Filter CSV to only include the requested columns
    cols_to_keep = ["incoming_email", "generated_reply", "overall_score"]
    csv_df = csv_df[[c for c in cols_to_keep if c in csv_df.columns]]

    for col in csv_df.columns:
        csv_df[col] = csv_df[col].apply(_serialize_for_csv)
    csv_df.to_csv(config["output_csv_path"], index=False)
    logger.info("Per-response results written to '%s'", config["output_csv_path"])

    report = {
        "summary": summary,
        "config": {k: v for k, v in config.items()},
        "results": json.loads(judged_df.to_json(orient="records")),
    }
    with open(config["output_json_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Full report written to '%s'", config["output_json_path"])


def main() -> None:
    judged_df = run_pipeline(CONFIG)
    summary = build_summary(judged_df)

    print("\n" + "=" * 60)
    print("CLOUDSYNC SUGGESTED-RESPONSE SYSTEM -- EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")

    save_reports(judged_df, summary, CONFIG)


if __name__ == "__main__":
    main()
