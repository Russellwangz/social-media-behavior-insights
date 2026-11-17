from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.schema import RecommendationItem, RetrievedRecommendation, UserProfile

SBERT_MODEL_NAME = "efederici/sentence-bert-base"


def _overlap_count(a: Iterable[str], b: Iterable[str]) -> int:
    set_a = {str(x).strip().lower() for x in a if str(x).strip()}
    set_b = {str(x).strip().lower() for x in b if str(x).strip()}
    return len(set_a.intersection(set_b))


def _base_filter(item: RecommendationItem, profile: UserProfile) -> bool:
    level_ok = item.addiction_range[0] <= profile.addiction_level <= item.addiction_range[1]
    persona_ok = profile.persona in item.persona_tags
    severity_ok = profile.severity in item.severity_tags

    # Keep broad shared resources while favoring precise filters.
    return level_ok and (persona_ok or severity_ok)


def _profile_to_text(profile: UserProfile) -> str:
    parts = [
        profile.persona,
        profile.severity,
        profile.interpretation,
        " ".join(profile.top_matching_features),
        " ".join(profile.prevention_ideas),
        " ".join(profile.risk_flags),
    ]
    return " ".join(x for x in parts if x).strip()


def _item_to_text(item: RecommendationItem) -> str:
    return " ".join(
        [
            item.title,
            item.content,
            " ".join(item.persona_tags),
            " ".join(item.feature_tags),
            " ".join(item.goal_tags),
            " ".join(item.severity_tags),
            f"difficulty {item.difficulty}",
        ]
    ).strip()


def _embedding_scores(
    query_text: str,
    items: list[RecommendationItem],
) -> np.ndarray:
    return _sbert_dense_scores(query_text, items)


@lru_cache(maxsize=1)
def _load_sbert_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SBERT_MODEL_NAME)


def _encode_with_sbert(texts: list[str]) -> np.ndarray:
    model = _load_sbert_encoder()
    return np.asarray(model.encode(texts))


def _sbert_dense_scores(
    query_text: str,
    items: list[RecommendationItem],
) -> np.ndarray:
    item_texts = [_item_to_text(item) for item in items]
    embeddings = _encode_with_sbert([query_text] + item_texts)
    query_vec = embeddings[0].reshape(1, -1)
    item_vecs = embeddings[1:]
    return cosine_similarity(query_vec, item_vecs).ravel()


def _build_reasons(item: RecommendationItem, profile: UserProfile, score: float) -> list[str]:
    reasons = [f"cosine similarity ({score:.3f})"]
    if profile.persona in item.persona_tags:
        reasons.append("persona match")
    if profile.severity in item.severity_tags:
        reasons.append("severity match")
    feature_overlap = _overlap_count(profile.top_matching_features, item.feature_tags)
    if feature_overlap:
        reasons.append(f"feature overlap ({feature_overlap})")
    return reasons


def retrieve_recommendations(
    profile: UserProfile,
    knowledge_base: list[RecommendationItem],
    top_k: int = 3,
) -> list[RetrievedRecommendation]:
    """Vector-embedding retrieval using cosine similarity."""
    candidates = [item for item in knowledge_base if _base_filter(item, profile)]
    if not candidates:
        candidates = knowledge_base

    query_text = _profile_to_text(profile)
    scored: list[RetrievedRecommendation] = []

    try:
        similarities = _embedding_scores(query_text, candidates)
        for item, score in zip(candidates, similarities):
            scored.append(
                RetrievedRecommendation(
                    item=item,
                    score=float(score),
                    reasons=_build_reasons(item, profile, float(score)),
                )
            )
    except Exception:
        # Fallback to deterministic rule-only scoring if vectorization fails.
        for item in candidates:
            persona_bonus = 1.0 if profile.persona in item.persona_tags else 0.0
            severity_bonus = 0.5 if profile.severity in item.severity_tags else 0.0
            feature_bonus = 0.1 * _overlap_count(profile.top_matching_features, item.feature_tags)
            score = persona_bonus + severity_bonus + feature_bonus
            scored.append(
                RetrievedRecommendation(
                    item=item,
                    score=float(score),
                    reasons=["fallback ranking (embedding unavailable)"],
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: max(3, min(top_k, 5))]
