"""Azure OpenAI helper for glossary alignment workflows."""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from .glossary_linker import CandidateMatch, ClassificationService, EmbeddingService, GlossaryTerm, SelectionResult


class AzureOpenAIService(EmbeddingService, ClassificationService):
    """Wrapper around Azure OpenAI for embeddings and chat completions."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_deployment: Optional[str] = None,
        chat_deployment: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import AzureOpenAI

        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        self._embedding_deployment = embedding_deployment or os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )
        self._chat_deployment = chat_deployment or os.environ.get(
            "AZURE_OPENAI_CHAT_DEPLOYMENT"
        )
        if not self._endpoint or not self._api_key or not self._api_version:
            raise EnvironmentError(
                "Azure OpenAI configuration missing. Ensure AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_VERSION are set."
            )
        if not self._embedding_deployment:
            raise EnvironmentError(
                "Embedding deployment name missing. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
            )
        if not self._chat_deployment:
            raise EnvironmentError(
                "Chat deployment name missing. Set AZURE_OPENAI_CHAT_DEPLOYMENT."
            )

        self._client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )
        self._temperature = float(temperature)

    # EmbeddingService -----------------------------------------------------
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        response = self._client.embeddings.create(
            model=self._embedding_deployment,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    # ClassificationService ------------------------------------------------
    def classify_top_class(
        self, term: GlossaryTerm, candidate_classes: Sequence[str]
    ) -> str:
        classes_text = ", ".join(candidate_classes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an ontology assistant. Classify glossary entries into "
                    "broad categories provided to you. Respond with exactly one of the "
                    "listed category labels."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Candidate classes: {classes_text}\n\n"
                    f"Glossary entry:\n{term.as_prompt()}\n\n"
                    "Answer with the single best matching class label."
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=self._chat_deployment,
            temperature=self._temperature,
            messages=messages,
        )
        content = (response.choices[0].message.content or "").strip()
        return self._match_class_label(content, candidate_classes)

    def select_best_candidate(
        self, term: GlossaryTerm, candidates: Sequence[CandidateMatch]
    ) -> SelectionResult:
        if not candidates:
            return SelectionResult(uri=None, rationale="No ontology candidates produced")

        candidate_lines = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_lines.append(
                f"{index}. URI: {candidate.uri}\n"
                f"   Label: {candidate.label}\n"
                f"   Similarity: {candidate.similarity:.3f}\n"
                f"   Details: {candidate.description or candidate.raw_text}"
            )
        payload = "\n".join(candidate_lines)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are assisting with ontology alignment. Choose the candidate that best "
                    "matches the glossary entry. Reply with the URI of the best match. If none "
                    "fit, reply with the word NONE."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Glossary entry:\n{term.as_prompt()}\n\n"
                    f"Candidates:\n{payload}\n\n"
                    "Return only the URI of the best match, or NONE."
                ),
            },
        ]
        response = self._client.chat.completions.create(
            model=self._chat_deployment,
            temperature=self._temperature,
            messages=messages,
        )
        content = (response.choices[0].message.content or "").strip()
        uri = self._extract_uri(content, candidates)
        return SelectionResult(uri=uri, rationale=content)

    # Helpers --------------------------------------------------------------
    @staticmethod
    def _match_class_label(content: str, candidate_classes: Sequence[str]) -> str:
        if not content:
            return candidate_classes[0]
        normalized = content.lower()
        for label in candidate_classes:
            if label.lower() in normalized:
                return label
        # Attempt to match by numeric index (1-based)
        digits = re.findall(r"\d+", normalized)
        for value in digits:
            index = int(value) - 1
            if 0 <= index < len(candidate_classes):
                return candidate_classes[index]
        return candidate_classes[0]

    @staticmethod
    def _extract_uri(content: str, candidates: Sequence[CandidateMatch]) -> Optional[str]:
        if not content:
            return candidates[0].uri if candidates else None
        lowered = content.lower()
        if "none" in lowered:
            return None
        for candidate in candidates:
            if candidate.uri in content:
                return candidate.uri
        digits = re.findall(r"\d+", content)
        for value in digits:
            index = int(value) - 1
            if 0 <= index < len(candidates):
                return candidates[index].uri
        return candidates[0].uri


__all__ = ["AzureOpenAIService"]
