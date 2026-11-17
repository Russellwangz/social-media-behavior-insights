from __future__ import annotations

from datetime import datetime
from typing import Any

from existing_predictors.addiction_predictor import severity_from_addiction_level
from models.schema import UserProfile


def _parse_hour_from_inputs(raw_inputs: dict[str, Any]) -> int | None:
    if raw_inputs.get("Watch Time Hour") is not None:
        return int(raw_inputs["Watch Time Hour"])

    watch_text = raw_inputs.get("Watch Time")
    if not watch_text:
        return None

    formats = ["%I:%M %p", "%I %p", "%H:%M", "%H"]
    for fmt in formats:
        try:
            return datetime.strptime(str(watch_text), fmt).hour
        except ValueError:
            continue
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def derive_risk_flags(
    raw_inputs: dict[str, Any],
    normalized_inputs: dict[str, Any],
    persona_name: str,
    addiction_level: int,
) -> list[str]:
    """Derive optional risk flags from raw questionnaire + normalized values."""
    flags: list[str] = []

    hour = _parse_hour_from_inputs(raw_inputs)
    if hour is not None and (hour >= 22 or hour <= 1):
        flags.append("late_night_usage")

    total_time = _safe_float(normalized_inputs.get("Total Time Spent"))
    if total_time is not None and total_time >= 180:
        flags.append("high_time_spent")

    sessions = _safe_float(normalized_inputs.get("Number of Sessions"))
    if sessions is not None and sessions >= 40:
        flags.append("high_session_count")

    engagement_text = str(raw_inputs.get("Engagement", "")).strip().upper()
    engagement_num = _safe_float(normalized_inputs.get("Engagement"))
    if engagement_text in {"D", "E"} or (engagement_num is not None and engagement_num >= 6000):
        flags.append("high_engagement")

    freq = str(raw_inputs.get("Frequency", "")).strip().lower()
    if persona_name == "Frequent Interruption Scrollers" or (
        sessions is not None and sessions >= 30 and freq in {"morning", "afternoon", "evening"}
    ):
        flags.append("study_interruption_risk")

    if addiction_level >= 8 and "high_time_spent" not in flags:
        flags.append("high_time_spent")

    return sorted(set(flags))


def _extract_top_feature_names(top_features: list[Any]) -> list[str]:
    names: list[str] = []
    for item in top_features:
        if isinstance(item, (list, tuple)) and item:
            names.append(str(item[0]))
        elif isinstance(item, str):
            names.append(item)
    return names


def build_user_profile(
    raw_inputs: dict[str, Any],
    persona_result: dict[str, Any],
    addiction_result: dict[str, Any],
) -> UserProfile:
    """Build a normalized user profile for recommendation retrieval."""
    persona_name = str(persona_result.get("Predicted Persona", "Unknown"))
    persona_score = _safe_float(persona_result.get("Best Score"))

    addiction_level = int(addiction_result.get("predicted_addiction_level", 5))
    severity = addiction_result.get("severity") or severity_from_addiction_level(addiction_level)

    top_features = _extract_top_feature_names(persona_result.get("Top Matching Features", []))
    prevention_ideas = [str(x) for x in persona_result.get("Suggested prevention ideas", [])]
    interpretation = str(persona_result.get("Interpretation", ""))

    normalized_inputs = {}
    normalized_inputs.update(persona_result.get("Normalized Features", {}))
    normalized_inputs.update(addiction_result.get("normalized_features", {}))

    risk_flags = derive_risk_flags(raw_inputs, normalized_inputs, persona_name, addiction_level)

    return UserProfile(
        persona=persona_name,
        persona_score=persona_score,
        addiction_level=addiction_level,
        severity=str(severity),
        top_matching_features=top_features,
        interpretation=interpretation,
        prevention_ideas=prevention_ideas,
        risk_flags=risk_flags,
        raw_inputs=raw_inputs,
        normalized_inputs=normalized_inputs,
    )
