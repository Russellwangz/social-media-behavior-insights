"""Compatibility wrapper for addiction prediction.

Primary implementation now lives in `existing_predictors/addiction_predictor.py`.
"""

from __future__ import annotations

from typing import Any

from existing_predictors.addiction_predictor import (
    format_addiction_prediction,
    predict_addiction_structured,
    predict_from_questionnaire,
)

__all__ = [
    "predict_from_questionnaire",
    "predict_addiction_structured",
    "format_addiction_prediction",
]


def main() -> None:
    sample: dict[str, Any] = {
        "Platform": "TikTok",
        "Total Time Spent": "1-2 hours",
        "Number of Sessions": "11-20",
        "Watch Time": "10:30 PM",
    }
    result = predict_addiction_structured(sample)
    print(format_addiction_prediction(result))


if __name__ == "__main__":
    main()
