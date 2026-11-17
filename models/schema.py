from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendationItem:
    """A single intervention recommendation in the local knowledge base."""

    id: str
    title: str
    persona_tags: list[str]
    addiction_range: tuple[int, int]
    severity_tags: list[str]
    feature_tags: list[str]
    goal_tags: list[str]
    difficulty: str
    content: str


@dataclass
class RetrievedRecommendation:
    """Recommendation with explainable retrieval metadata."""

    item: RecommendationItem
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """Normalized profile combining model outputs and risk flags."""

    persona: str
    persona_score: float | None
    addiction_level: int
    severity: str
    top_matching_features: list[str]
    interpretation: str
    prevention_ideas: list[str]
    risk_flags: list[str]
    raw_inputs: dict[str, Any]
    normalized_inputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "persona_score": self.persona_score,
            "addiction_level": self.addiction_level,
            "severity": self.severity,
            "top_matching_features": self.top_matching_features,
            "interpretation": self.interpretation,
            "prevention_ideas": self.prevention_ideas,
            "risk_flags": self.risk_flags,
            "raw_inputs": self.raw_inputs,
            "normalized_inputs": self.normalized_inputs,
        }
