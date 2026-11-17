"""
Persona-aware recommendation prototype (local, no external APIs).

Run:
    python app.py
"""

from __future__ import annotations

import json

from data.sample_inputs import get_sample_questionnaire
from existing_predictors.addiction_predictor import (
    format_addiction_prediction,
    predict_addiction_structured,
)
from existing_predictors.persona_classifier import classify_persona
from generator import generate_recommendation_response
from knowledge_base import load_knowledge_base
from profile_builder import build_user_profile
from retriever import retrieve_recommendations


def run_pipeline(raw_input: dict) -> dict:
    persona_result = classify_persona(raw_input)
    addiction_result = predict_addiction_structured(raw_input)
    profile = build_user_profile(raw_input, persona_result, addiction_result)

    kb = load_knowledge_base()
    retrieved = retrieve_recommendations(profile, kb, top_k=5)
    final_response = generate_recommendation_response(profile, retrieved)

    return {
        "persona_result": persona_result,
        "addiction_result": addiction_result,
        "profile": profile.to_dict(),
        "retrieved": [
            {
                "id": r.item.id,
                "title": r.item.title,
                "score": round(r.score, 3),
                "reasons": r.reasons,
            }
            for r in retrieved
        ],
        "final_response": final_response,
    }


def main() -> None:
    raw_input = get_sample_questionnaire()
    output = run_pipeline(raw_input)

    print("=== Raw Input ===")
    print(json.dumps(raw_input, indent=2))
    print()

    print("=== Persona Prediction ===")
    print(json.dumps(output["persona_result"], indent=2, default=str))
    print()

    print("=== Addiction Prediction ===")
    print(format_addiction_prediction(output["addiction_result"]))
    print(json.dumps(output["addiction_result"], indent=2, default=str))
    print()

    print("=== Structured User Profile ===")
    print(json.dumps(output["profile"], indent=2, default=str))
    print()

    print("=== Retrieved Recommendations ===")
    print(json.dumps(output["retrieved"], indent=2))
    print()

    print("=== Final Personalized Recommendation ===")
    print(output["final_response"])


if __name__ == "__main__":
    main()
