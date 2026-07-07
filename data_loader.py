"""
data_loader.py
==============

Loads the three CloudSync customer-support datasets (produced by Copilot,
Gemini, and Claude respectively), validates and merges them into a single
in-memory representation, and splits the merged data into:

  - a "Knowledge Base" (used for Retrieval-Augmented Generation few-shot
    retrieval), and
  - a "Test Set" (used for generation + LLM-as-a-Judge evaluation).

Each source file is expected to be either:
  - a JSON file containing a list of record dicts, or
  - a JSON file containing a top-level dict with a list under a key such as
    "data" / "records" / "examples", or
  - a newline-delimited JSON (.jsonl / .txt) file, one record per line.

Each record is expected to contain (at minimum):
    id                  -> unique identifier (str/int)
    intent               -> short label for the customer's intent
    incoming_email        -> the customer's incoming support email (str)
    ground_truth_reply    -> the "gold" reply written by a human/expert (str)
    gold_requirements      -> list[str] or str describing what a correct
                             reply must accomplish
    must_include          -> list[str] of phrases/facts that should appear
    must_not_include      -> list[str] of phrases/facts that must NOT appear

Records missing required fields are logged and dropped (not silently
ignored), so data quality issues surface immediately instead of causing
confusing downstream failures.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "id",
    "intent",
    "incoming_email",
    "ground_truth_reply",
    "gold_requirements",
    "must_include",
    "must_not_include",
]

# Keys under which a list of records might be nested if the top-level JSON
# object is a dict rather than a list.
POSSIBLE_LIST_KEYS = ["data", "records", "examples", "items", "conversations"]


def _normalize_list_field(value: Any) -> List[str]:
    """Coerce must_include / must_not_include / gold_requirements into a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        # Some source files may store these as a single comma or
        # newline separated string rather than a JSON list.
        parts = [p.strip() for p in value.replace("\n", ",").split(",")]
        return [p for p in parts if p]
    # Fallback: stringify whatever it is.
    return [str(value)]


def _extract_records(raw: Any, source_name: str) -> List[Dict[str, Any]]:
    """Given parsed JSON (list or dict), extract the list of record dicts."""
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in POSSIBLE_LIST_KEYS:
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        # Maybe the dict itself IS a single record.
        if all(f in raw for f in ("incoming_email", "ground_truth_reply")):
            return [raw]

    raise ValueError(
        f"Could not locate a list of records in '{source_name}'. "
        f"Expected a JSON list, or a dict containing one of {POSSIBLE_LIST_KEYS}."
    )


def _load_single_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load one dataset file (.json, .jsonl, or .txt) into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("Dataset file '%s' is empty.", path)
        return []

    # Try parsing as a single JSON document first (list or dict).
    try:
        raw = json.loads(text)
        return _extract_records(raw, str(path))
    except json.JSONDecodeError:
        pass

    # Fall back to a streaming multi-document parse. This handles:
    #   - true JSONL (one compact JSON object per line)
    #   - pretty-printed / indented JSON objects concatenated back-to-back
    #     (each record spans multiple lines) -- this is what tripped up a
    #     naive "one json.loads() per line" approach, since splitting a
    #     pretty-printed object on newlines shreds it into fragments like
    #     `    "id": "123",` which are not valid JSON on their own.
    #   - the same, wrapped in an outer `[ ... ]` with trailing/leading
    #     commas or stray brackets that break whole-file parsing.
    decoder = json.JSONDecoder()
    records: List[Dict[str, Any]] = []
    idx = 0
    n = len(text)
    skipped_chars = 0

    while idx < n:
        # Skip whitespace, commas, and stray array brackets between records.
        while idx < n and text[idx] in " \t\r\n,[]":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            records.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            # Could not decode a full object starting here -- advance one
            # character and keep trying rather than giving up on the whole
            # file. Track how much we skip so we can warn if it's a lot.
            skipped_chars += 1
            idx += 1

    if skipped_chars:
        logger.warning(
            "Skipped %d unparsable character(s) while streaming JSON objects "
            "from '%s' (this is normal for a few stray separators; if it's a "
            "large number, the file may be malformed).",
            skipped_chars, path,
        )

    if not records:
        raise ValueError(
            f"File '{path}' is neither a single valid JSON document nor a "
            f"parsable sequence of JSON objects."
        )

    logger.info("Parsed %d JSON object(s) from '%s' via streaming decode.", len(records), path)

    # If the "objects" we decoded are themselves lists (e.g. the whole file
    # was one array but json.loads() on the full text failed due to a single
    # trailing comma), flatten one level.
    flattened: List[Dict[str, Any]] = []
    for obj in records:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)
    return flattened


def _validate_and_clean_record(
    record: Dict[str, Any], source_name: str, idx: int
) -> Optional[Dict[str, Any]]:
    """Validate a single record; return a cleaned copy, or None if invalid."""
    if not isinstance(record, dict):
        logger.warning("Dropping record #%d from '%s' (not a JSON object)", idx, source_name)
        return None

    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        logger.warning(
            "Dropping record #%d from '%s' (missing fields: %s)",
            idx, source_name, missing,
        )
        return None

    incoming_email = str(record.get("incoming_email", "")).strip()
    ground_truth_reply = str(record.get("ground_truth_reply", "")).strip()
    if not incoming_email or not ground_truth_reply:
        logger.warning(
            "Dropping record #%d from '%s' (empty incoming_email or ground_truth_reply)",
            idx, source_name,
        )
        return None

    cleaned = {
        "id": str(record["id"]),
        "source": source_name,
        "intent": str(record.get("intent", "")).strip() or "unknown",
        "incoming_email": incoming_email,
        "ground_truth_reply": ground_truth_reply,
        "gold_requirements": _normalize_list_field(record.get("gold_requirements")),
        "must_include": _normalize_list_field(record.get("must_include")),
        "must_not_include": _normalize_list_field(record.get("must_not_include")),
    }
    return cleaned


@dataclass
class MergedDataset:
    """Container for the merged and split dataset."""

    full_df: pd.DataFrame
    knowledge_base_df: pd.DataFrame
    test_df: pd.DataFrame
    stats: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Dataset summary",
            "----------------",
            f"Total valid records : {len(self.full_df)}",
            f"Knowledge base size  : {len(self.knowledge_base_df)}",
            f"Test set size        : {len(self.test_df)}",
        ]
        if "records_per_source" in self.stats:
            lines.append(f"Records per source    : {self.stats['records_per_source']}")
        if "dropped_records" in self.stats:
            lines.append(f"Dropped records       : {self.stats['dropped_records']}")
        return "\n".join(lines)


def load_and_merge_datasets(
    file_paths: List[Union[str, Path]],
    source_names: Optional[List[str]] = None,
    dedupe_on: str = "incoming_email",
) -> pd.DataFrame:
    """
    Load one or more dataset files and merge them into a single DataFrame.

    Args:
        file_paths: paths to the dataset files (e.g. copilot.json, gemini.json,
            claude.json).
        source_names: optional human-readable labels (defaults to file stems).
        dedupe_on: column used to drop exact duplicate records that may occur
            across the three source files (default: incoming_email text).

    Returns:
        A pandas DataFrame of cleaned, validated records with a "source" column.
    """
    if source_names is None:
        source_names = [Path(p).stem for p in file_paths]
    if len(source_names) != len(file_paths):
        raise ValueError("source_names must be the same length as file_paths")

    all_records: List[Dict[str, Any]] = []
    records_per_source: Dict[str, int] = {}
    dropped = 0

    for path, name in zip(file_paths, source_names):
        try:
            raw_records = _load_single_file(path)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Failed to load dataset '%s': %s", path, e)
            continue

        valid_count = 0
        for idx, rec in enumerate(raw_records):
            cleaned = _validate_and_clean_record(rec, name, idx)
            if cleaned is not None:
                all_records.append(cleaned)
                valid_count += 1
            else:
                dropped += 1

        records_per_source[name] = valid_count
        logger.info("Loaded %d valid records from '%s' (%s)", valid_count, path, name)

    if not all_records:
        raise RuntimeError(
            "No valid records were loaded from any of the provided dataset files. "
            "Check file paths and formats."
        )

    df = pd.DataFrame(all_records)

    before = len(df)
    if dedupe_on in df.columns:
        df = df.drop_duplicates(subset=[dedupe_on]).reset_index(drop=True)
    after = len(df)
    if before != after:
        logger.info("Dropped %d duplicate records (matched on '%s')", before - after, dedupe_on)

    df.attrs["records_per_source"] = records_per_source
    df.attrs["dropped_records"] = dropped
    return df


def split_knowledge_base_and_test_set(
    df: pd.DataFrame,
    test_fraction: float = 0.3,
    random_seed: int = 42,
    stratify_on_intent: bool = True,
) -> MergedDataset:
    """
    Split the merged DataFrame into a knowledge base (for RAG retrieval) and
    a test set (for generation + evaluation).

    Args:
        df: merged DataFrame from load_and_merge_datasets().
        test_fraction: fraction of records to hold out as the test set.
        random_seed: seed for reproducibility.
        stratify_on_intent: if True, attempts to split proportionally within
            each 'intent' group so both splits cover the same intents.

    Returns:
        A MergedDataset dataclass with full_df, knowledge_base_df, test_df.
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0 and 1 (exclusive)")

    rng = random.Random(random_seed)
    test_indices: List[int] = []

    if stratify_on_intent and "intent" in df.columns:
        for _, group in df.groupby("intent"):
            idxs = list(group.index)
            rng.shuffle(idxs)
            n_test = max(1, round(len(idxs) * test_fraction)) if len(idxs) > 1 else (
                1 if rng.random() < test_fraction else 0
            )
            test_indices.extend(idxs[:n_test])
    else:
        idxs = list(df.index)
        rng.shuffle(idxs)
        n_test = max(1, round(len(idxs) * test_fraction))
        test_indices = idxs[:n_test]

    test_mask = df.index.isin(test_indices)
    test_df = df[test_mask].reset_index(drop=True)
    kb_df = df[~test_mask].reset_index(drop=True)

    if len(kb_df) == 0:
        raise RuntimeError(
            "Knowledge base ended up empty after splitting -- lower test_fraction "
            "or provide more data."
        )
    if len(test_df) == 0:
        raise RuntimeError(
            "Test set ended up empty after splitting -- raise test_fraction "
            "or provide more data."
        )

    stats = {
        "records_per_source": df.attrs.get("records_per_source", {}),
        "dropped_records": df.attrs.get("dropped_records", 0),
    }

    return MergedDataset(full_df=df, knowledge_base_df=kb_df, test_df=test_df, stats=stats)


def load_dataset_bundle(
    file_paths: List[Union[str, Path]],
    source_names: Optional[List[str]] = None,
    test_fraction: float = 0.3,
    random_seed: int = 42,
) -> MergedDataset:
    """Convenience wrapper: load, merge, and split in one call."""
    df = load_and_merge_datasets(file_paths, source_names=source_names)
    return split_knowledge_base_and_test_set(
        df, test_fraction=test_fraction, random_seed=random_seed
    )


if __name__ == "__main__":
    # Example usage (adjust paths to your actual dataset filenames).
    example_paths = [
        "datasets/copilot_dataset.json",
        "datasets/gemini_dataset.json",
        "datasets/claude_dataset.json",
    ]
    try:
        bundle = load_dataset_bundle(example_paths)
        print(bundle.summary())
    except Exception as exc:
        logger.error("Dataset loading failed: %s", exc)
