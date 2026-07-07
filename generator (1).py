"""
generator.py
============

Retrieval-Augmented Generation (RAG) pipeline for producing suggested email
replies.

Pipeline:
  1. Embed every "Knowledge Base" record (incoming_email + ground_truth_reply)
     using a sentence-transformers model (default: all-MiniLM-L6-v2).
  2. For a new incoming email from the Test Set, embed it and retrieve the
     Top-K most similar knowledge-base records via cosine similarity.
  3. Build a few-shot prompt containing those retrieved examples plus the
     new email, and pass it to an `LLMProvider` (Strategy Pattern) to
     generate the suggested reply.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from llm_provider import LLMProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a senior customer support agent for a company called CloudSync, "
    "a cloud file-sync and backup product. Write clear, professional, and "
    "empathetic email replies that directly resolve the customer's issue. "
    "Match the tone and structure of the example replies you are given, but "
    "do not copy them verbatim -- tailor your reply to the new customer's "
    "specific email."
)


@dataclass
class RetrievedExample:
    incoming_email: str
    ground_truth_reply: str
    similarity: float
    intent: str


class RAGGenerator:
    """
    Encapsulates the embedding index over the knowledge base and the
    retrieval + prompt-construction + generation logic.
    """

    def __init__(
        self,
        knowledge_base_df: pd.DataFrame,
        llm_provider: LLMProvider,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 3,
    ):
        if knowledge_base_df.empty:
            raise ValueError("knowledge_base_df must contain at least one record.")
        required_cols = {"incoming_email", "ground_truth_reply", "intent"}
        missing = required_cols - set(knowledge_base_df.columns)
        if missing:
            raise ValueError(f"knowledge_base_df is missing required columns: {missing}")

        self.knowledge_base_df = knowledge_base_df.reset_index(drop=True)
        self.llm_provider = llm_provider
        self.top_k = top_k

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for RAGGenerator. "
                "Install it with `pip install sentence-transformers`."
            ) from e

        logger.info("Loading embedding model '%s'...", embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        logger.info(
            "Embedding %d knowledge-base records...", len(self.knowledge_base_df)
        )
        # Embed the incoming_email text only, since that's what we match a
        # NEW incoming email against at inference time.
        self.kb_embeddings = self.embedding_model.encode(
            self.knowledge_base_df["incoming_email"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def retrieve(self, incoming_email: str, top_k: Optional[int] = None) -> List[RetrievedExample]:
        """Retrieve the top-k most similar knowledge-base examples for a new email."""
        top_k = top_k or self.top_k
        top_k = min(top_k, len(self.knowledge_base_df))

        query_embedding = self.embedding_model.encode(
            [incoming_email], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        # Embeddings are normalized, so dot product == cosine similarity.
        similarities = self.kb_embeddings @ query_embedding
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            row = self.knowledge_base_df.iloc[idx]
            results.append(
                RetrievedExample(
                    incoming_email=row["incoming_email"],
                    ground_truth_reply=row["ground_truth_reply"],
                    similarity=float(similarities[idx]),
                    intent=row.get("intent", "unknown"),
                )
            )
        return results

    def _build_prompt(self, incoming_email: str, examples: List[RetrievedExample]) -> str:
        blocks = []
        for i, ex in enumerate(examples, start=1):
            blocks.append(
                f"--- Example {i} (intent: {ex.intent}, similarity: {ex.similarity:.2f}) ---\n"
                f"Customer email:\n{ex.incoming_email}\n\n"
                f"Agent reply:\n{ex.ground_truth_reply}\n"
            )
        examples_block = "\n".join(blocks) if blocks else "(no similar examples found)"

        prompt = (
            "Below are past customer support emails and the replies our agents sent. "
            "Use them as style/content references.\n\n"
            f"{examples_block}\n"
            "--- New customer email to answer ---\n"
            f"{incoming_email}\n\n"
            "Write the best possible reply to this new customer email. "
            "Respond with ONLY the reply text (no preamble, no explanation)."
        )
        return prompt

    def generate_reply(
        self,
        incoming_email: str,
        top_k: Optional[int] = None,
        temperature: float = 0.4,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Retrieve few-shot examples and generate a suggested reply for a new
        incoming email.

        Returns a dict with the generated reply text plus retrieval metadata
        (useful for debugging / auditing which examples influenced the reply).
        """
        examples = self.retrieve(incoming_email, top_k=top_k)
        prompt = self._build_prompt(incoming_email, examples)

        try:
            reply_text = self.llm_provider.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Generation failed for email (first 60 chars: %r): %s",
                         incoming_email[:60], exc)
            reply_text = ""

        return {
            "generated_reply": reply_text,
            "retrieved_examples": [
                {
                    "incoming_email": ex.incoming_email,
                    "similarity": ex.similarity,
                    "intent": ex.intent,
                }
                for ex in examples
            ],
        }

    def generate_replies_for_test_set(
        self,
        test_df: pd.DataFrame,
        top_k: Optional[int] = None,
        temperature: float = 0.4,
        max_tokens: int = 500,
    ) -> pd.DataFrame:
        """Generate replies for every row in a test-set DataFrame."""
        if "incoming_email" not in test_df.columns:
            raise ValueError("test_df must contain an 'incoming_email' column.")

        rows = []
        for i, row in test_df.reset_index(drop=True).iterrows():
            logger.info("Generating reply %d/%d (id=%s)...", i + 1, len(test_df), row.get("id", i))
            result = self.generate_reply(
                incoming_email=row["incoming_email"],
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            merged = row.to_dict()
            merged["generated_reply"] = result["generated_reply"]
            merged["retrieved_examples"] = result["retrieved_examples"]
            rows.append(merged)

        return pd.DataFrame(rows)
