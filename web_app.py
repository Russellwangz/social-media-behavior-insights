"""
Flask web application for Internet Addiction Assessment.

Run:
    python web_app.py
Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import json
from flask import Flask, render_template, request, jsonify

from existing_predictors.persona_classifier import classify_persona
from existing_predictors.addiction_predictor import (
    predict_addiction_structured,
    format_addiction_prediction,
)
from profile_builder import build_user_profile
from knowledge_base import load_knowledge_base
from retriever import retrieve_recommendations
from generator import generate_recommendation_response

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True)

    raw_input = _build_raw_input(data)

    persona_result = classify_persona(raw_input)

    try:
        addiction_result = predict_addiction_structured(raw_input)
    except Exception:
        addiction_result = {
            "predicted_addiction_level": 5,
            "severity": "medium",
            "normalized_features": {},
            "behavior_summary": {},
            "model_name": "unavailable",
        }

    profile = build_user_profile(raw_input, persona_result, addiction_result)

    kb = load_knowledge_base()
    retrieved = retrieve_recommendations(profile, kb, top_k=5)
    final_response = generate_recommendation_response(profile, retrieved)

    return jsonify({
        "persona": {
            "name": persona_result["Predicted Persona"],
            "score": persona_result["Best Score"],
            "all_scores": persona_result["All Scores"],
            "top_features": [
                {"feature": f[0], "score": round(f[1], 4)}
                for f in persona_result["Top Matching Features"]
            ],
            "interpretation": persona_result["Interpretation"],
            "prevention_ideas": persona_result["Suggested prevention ideas"],
        },
        "addiction": {
            "level": addiction_result["predicted_addiction_level"],
            "severity": addiction_result["severity"],
            "model": addiction_result["model_name"],
        },
        "profile": profile.to_dict(),
        "recommendations": [
            {
                "id": r.item.id,
                "title": r.item.title,
                "content": r.item.content,
                "difficulty": r.item.difficulty,
                "score": round(r.score, 3),
                "reasons": r.reasons,
            }
            for r in retrieved
        ],
        "final_response": final_response,
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    original_input = _build_raw_input(data)

    TOTAL_TIME_TIERS = [
        "Less than 30 minutes", "30-60 minutes", "1-2 hours",
        "2-3 hours", "3-4 hours", "4-6 hours", "6-8 hours", "More than 8 hours",
    ]
    SESSIONS_TIERS = ["1-3", "4-6", "7-10", "11-15", "More than 15"]
    SAFE_PLATFORMS = {"YouTube", "Facebook"}
    RISKY_PLATFORMS = {"TikTok", "Instagram"}
    EARLIER_WATCH_TIME = "14:00"  # 2 PM — avoids late-night flag

    scenarios = []

    def run_scenario(label, description, modified_input):
        try:
            result = predict_addiction_structured(modified_input)
            return {
                "label": label,
                "description": description,
                "predicted_level": result["predicted_addiction_level"],
                "severity": result["severity"],
            }
        except Exception:
            return None

    # Scenario 1: Reduce total time by one tier
    current_time = data.get("total_time_spent", "1-2 hours")
    if current_time in TOTAL_TIME_TIERS:
        idx = TOTAL_TIME_TIERS.index(current_time)
        if idx > 0:
            reduced_time = TOTAL_TIME_TIERS[idx - 1]
            modified = {**original_input, "Total Time Spent": reduced_time}
            s = run_scenario(
                "Spend Less Time",
                f"Reduce daily usage from \"{current_time}\" to \"{reduced_time}\"",
                modified,
            )
            if s:
                scenarios.append(s)

    # Scenario 2: Reduce sessions by one tier
    current_sessions = data.get("num_sessions", "7-10")
    if current_sessions in SESSIONS_TIERS:
        idx = SESSIONS_TIERS.index(current_sessions)
        if idx > 0:
            reduced_sessions = SESSIONS_TIERS[idx - 1]
            modified = {**original_input, "Number of Sessions": reduced_sessions}
            s = run_scenario(
                "Open App Less Often",
                f"Cut daily sessions from \"{current_sessions}\" to \"{reduced_sessions}\"",
                modified,
            )
            if s:
                scenarios.append(s)

    # Scenario 3: Switch to earlier watch time
    current_hour = int(data.get("watch_time_hour", 14))
    if current_hour >= 21 or current_hour <= 2:
        modified = {**original_input, "Watch Time": EARLIER_WATCH_TIME}
        s = run_scenario(
            "Avoid Late-Night Use",
            "Shift your usage to the afternoon instead of late at night",
            modified,
        )
        if s:
            scenarios.append(s)

    # Scenario 4: Switch to lower-risk platform
    current_platform = data.get("platform", "YouTube")
    if current_platform in RISKY_PLATFORMS:
        alt_platform = "YouTube"
        modified = {**original_input, "Platform": alt_platform}
        s = run_scenario(
            f"Switch to {alt_platform}",
            f"Use {alt_platform} instead of {current_platform}",
            modified,
        )
        if s:
            scenarios.append(s)

    # Always include: combine all improvements
    best_time = TOTAL_TIME_TIERS[max(0, TOTAL_TIME_TIERS.index(current_time) - 1)] if current_time in TOTAL_TIME_TIERS else current_time
    best_sessions = SESSIONS_TIERS[max(0, SESSIONS_TIERS.index(current_sessions) - 1)] if current_sessions in SESSIONS_TIERS else current_sessions
    best_platform = "YouTube" if data.get("platform") in RISKY_PLATFORMS else data.get("platform", "YouTube")
    best_watch_time = EARLIER_WATCH_TIME if (current_hour >= 21 or current_hour <= 2) else original_input["Watch Time"]
    combined = {
        **original_input,
        "Total Time Spent": best_time,
        "Number of Sessions": best_sessions,
        "Platform": best_platform,
        "Watch Time": best_watch_time,
    }
    s = run_scenario(
        "All Changes Combined",
        "Apply all the improvements above at once",
        combined,
    )
    if s:
        scenarios.append(s)

    try:
        original_result = predict_addiction_structured(original_input)
        original_level = original_result["predicted_addiction_level"]
    except Exception:
        original_level = data.get("current_level", 5)

    return jsonify({
        "original_level": original_level,
        "scenarios": scenarios,
    })


def _build_raw_input(data: dict) -> dict:
    """Transform the frontend form payload into the dict format expected by the pipeline."""
    raw: dict = {}

    raw["Platform"] = data.get("platform", "YouTube")
    raw["Frequency"] = data.get("frequency", "Evening")
    raw["Video Category"] = data.get("video_category", "Jokes/Memes")

    hour = int(data.get("watch_time_hour", 14))
    minute = int(data.get("watch_time_minute", 0))
    raw["Watch Time"] = f"{hour:02d}:{minute:02d}"

    raw["Video Length"] = data.get("video_length", "C")
    raw["Total Time Spent"] = data.get("total_time_spent", "1-2 hours")
    raw["Number of Videos Watched"] = data.get("num_videos", "26-50")
    raw["Number of Sessions"] = data.get("num_sessions", "7-10")
    raw["Importance Score"] = str(data.get("importance_score", "5"))
    raw["Time Spent On Video"] = data.get("time_on_video", "C")
    raw["Scroll Rate"] = data.get("scroll_rate", "C")
    raw["Engagement"] = data.get("engagement", "C")
    raw["Overall Engagement Frequency"] = data.get("overall_engagement", "Sometimes")

    return raw


if __name__ == "__main__":
    import os
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
