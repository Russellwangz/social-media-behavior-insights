from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ADDICTION_FEATURES = [
    "Platform",
    "Total Time Spent",
    "Number of Sessions",
    "Watch Time",
]

ADDICTION_MODEL_CANDIDATES = [
    ROOT / "final_models" / "set_m_platform_pattern_addiction_level_logreg.joblib",
]

TOTAL_TIME_SPENT_MAP = {
    "Less than 30 minutes": 26.4375,
    "30-60 minutes": 60.0,
    "1-2 hours": 96.0,
    "2-3 hours": 131.0625,
    "3-4 hours": 171.0,
    "4-6 hours": 206.8125,
    "6-8 hours": 240.0,
    "More than 8 hours": 281.5625,
}

NUMBER_OF_SESSIONS_MAP = {
    "1-3": 2.0,
    "4-6": 7.0,
    "7-10": 10.0,
    "11-15": 14.0,
    "More than 15": 17.0,
}

WATCH_TIME_LABELS = [
    "10:15 PM",
    "11:30 PM",
    "2:00 PM",
    "3:45 PM",
    "3:55 PM",
    "4:25 PM",
    "5:00 PM",
    "5:45 PM",
    "6:05 PM",
    "7:25 PM",
    "7:45 AM",
    "8:00 AM",
    "8:30 PM",
    "9:00 PM",
    "9:15 AM",
    "9:55 AM",
]


def _norm_text(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace(" to ", "-")
        .replace("  ", " ")
    )


def _to_int_if_possible(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value.ravel()[0]
    try:
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    except Exception:
        return value


def _load_model(model_path: Path):
    import joblib

    return joblib.load(model_path)


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No model file found in candidates: {candidates}")


def _parse_user_time(value: str) -> datetime:
    text = value.strip()
    try:
        return datetime.strptime(text, "%H:%M")
    except ValueError as exc:
        raise ValueError("Unsupported Watch Time format. Use 24-hour 'HH:MM', e.g. '22:15'.") from exc


def _parse_seen_time_label(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%I:%M %p")


def _nearest_seen_time_label(user_time_text: str, seen_time_labels: list[str]) -> str:
    user_dt = _parse_user_time(user_time_text)

    parsed_seen: list[tuple[str, datetime]] = []
    for label in seen_time_labels:
        try:
            parsed_seen.append((label, _parse_seen_time_label(label)))
        except ValueError:
            continue

    if not parsed_seen:
        raise ValueError("No valid training watch-time labels found in dataset.")

    def _distance_sec(a: datetime, b: datetime) -> float:
        sec = abs((a - b).total_seconds())
        return min(sec, 24 * 3600 - sec)

    best_label, _ = min(parsed_seen, key=lambda x: _distance_sec(user_dt, x[1]))
    return best_label


def _idx(options: list[str], answer: str, aliases: dict[str, str] | None = None) -> int:
    normalized = _norm_text(answer)
    if aliases and normalized in aliases:
        normalized = aliases[normalized]
    for i, opt in enumerate(options):
        if normalized == _norm_text(opt):
            return i
    raise ValueError(f"Unsupported option: {answer!r}. Expected one of {options}.")


def _normalize_questionnaire_input(user_input: dict[str, Any]) -> dict[str, Any]:
    total_time_opts = list(TOTAL_TIME_SPENT_MAP.keys())
    n_sessions_opts = list(NUMBER_OF_SESSIONS_MAP.keys())

    total_idx = _idx(total_time_opts, str(user_input["Total Time Spent"]))
    sessions_idx = _idx(n_sessions_opts, str(user_input["Number of Sessions"]))
    watch_time_value = _nearest_seen_time_label(str(user_input["Watch Time"]), WATCH_TIME_LABELS)

    platform_map = {
        "tiktok": "TikTok",
        "instagram": "Instagram",
        "youtube": "YouTube",
        "facebook": "Facebook",
    }

    platform_key = _norm_text(str(user_input["Platform"]))
    if platform_key not in platform_map:
        raise ValueError("Unsupported Platform. Use one of: TikTok, Instagram, YouTube, Facebook.")

    total_time_value = TOTAL_TIME_SPENT_MAP[total_time_opts[total_idx]]
    sessions_value = NUMBER_OF_SESSIONS_MAP[n_sessions_opts[sessions_idx]]

    return {
        "Platform": platform_map[platform_key],
        "Total Time Spent": float(total_time_value),
        "Number of Sessions": float(sessions_value),
        "Watch Time": watch_time_value,
    }


def _calculate_behavior_summary(normalized: dict[str, Any]) -> dict[str, Any]:
    watch_dt = _parse_seen_time_label(str(normalized["Watch Time"]))
    hour = watch_dt.hour

    time_val = float(normalized["Total Time Spent"])
    sessions_val = float(normalized["Number of Sessions"])

    late_night_usage = hour >= 22 or hour <= 1
    high_time_spent = time_val >= TOTAL_TIME_SPENT_MAP["4-6 hours"]
    high_session_count = sessions_val >= NUMBER_OF_SESSIONS_MAP["11-15"]
    high_risk_platform = normalized["Platform"] in {"TikTok", "Instagram"}

    behavior_score = 0
    behavior_score += 2 if high_time_spent else 0
    behavior_score += 2 if high_session_count else 0
    behavior_score += 1 if late_night_usage else 0
    behavior_score += 1 if high_risk_platform else 0

    return {
        "behavior_score": behavior_score,
        "late_night_usage": late_night_usage,
        "high_time_spent": high_time_spent,
        "high_session_count": high_session_count,
        "high_risk_platform": high_risk_platform,
    }


def severity_from_addiction_level(level: int) -> str:
    if level <= 3:
        return "low"
    if level <= 6:
        return "medium"
    return "high"


def predict_addiction_structured(user_input: dict[str, Any]) -> dict[str, Any]:
    """Predict addiction level with Set_M logreg and return structured output."""
    normalized = _normalize_questionnaire_input(user_input)

    model_path = _resolve_existing_path(ADDICTION_MODEL_CANDIDATES)
    model = _load_model(model_path)

    X = pd.DataFrame([{f: normalized[f] for f in ADDICTION_FEATURES}], columns=ADDICTION_FEATURES)
    pred = _to_int_if_possible(np.asarray(model.predict(X)).ravel()[0])
    level = max(1, min(10, int(pred)))

    return {
        "predicted_addiction_level": level,
        "severity": severity_from_addiction_level(level),
        "normalized_features": normalized,
        "behavior_summary": _calculate_behavior_summary(normalized),
        "model_name": model_path.name,
    }


def format_addiction_prediction(result: dict[str, Any]) -> str:
    level = result["predicted_addiction_level"]
    model_name = result.get("model_name", "unknown")
    severity = result.get("severity", "unknown")
    return f"Predicted Addiction Level ({model_name}): {level} (severity: {severity})."


def predict_from_questionnaire(user_input: dict[str, Any]) -> str:
    """
    Backward-compatible wrapper returning display string.
    Required fields: Platform, Total Time Spent, Number of Sessions, Watch Time.
    """
    return format_addiction_prediction(predict_addiction_structured(user_input))


def get_addiction_questionnaire_options() -> dict[str, list[str]]:
    return {
        "Platform": ["TikTok", "Instagram", "YouTube", "Facebook"],
        "Total Time Spent": list(TOTAL_TIME_SPENT_MAP.keys()),
        "Number of Sessions": list(NUMBER_OF_SESSIONS_MAP.keys()),
        "Watch Time labels": WATCH_TIME_LABELS[:],
        "Watch Time input format": ["00:30", "09:15", "14:35", "22:15"],
    }
