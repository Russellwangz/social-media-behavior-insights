from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional


# =========================================================
# 1. Persona prototype definition
# =========================================================

@dataclass
class PersonaPrototype:
    name: str
    numeric_means: Dict[str, float]
    categorical_modes: Dict[str, str]


PERSONA_PROTOTYPES = [
    PersonaPrototype(
        name="Long-Session Heavy Viewers",
        numeric_means={
            "Total Time Spent": 173.79,
            "Number of Sessions": 9.88,
            "Video Length": 18.06,
            "Time Spent On Video": 15.14,
            "Number of Videos Watched": 34.58,
            "Scroll Rate": 45.55,
            "Engagement": 4002.36,
            "Importance Score": 3.52,
        },
        categorical_modes={
            "Platform": "YouTube",
            "Video Category": "Jokes/Memes",
            "Frequency": "Evening",
            "Watch Time": "Afternoon",
        },
    ),
    PersonaPrototype(
        name="High-Engagement Night Habit Users",
        numeric_means={
            "Total Time Spent": 117.50,
            "Number of Sessions": 8.34,
            "Video Length": 15.31,
            "Time Spent On Video": 11.04,
            "Number of Videos Watched": 26.70,
            "Scroll Rate": 50.46,
            "Engagement": 6476.18,
            "Importance Score": 7.10,
        },
        categorical_modes={
            "Platform": "Instagram",
            "Video Category": "Life Hacks",
            "Frequency": "Night",
            "Watch Time": "Night",
        },
    ),
    PersonaPrototype(
        name="Frequent Interruption Scrollers",
        numeric_means={
            "Total Time Spent": 160.13,
            "Number of Sessions": 11.86,
            "Video Length": 11.84,
            "Time Spent On Video": 18.78,
            "Number of Videos Watched": 13.06,
            "Scroll Rate": 53.94,
            "Engagement": 4637.50,
            "Importance Score": 4.97,
        },
        categorical_modes={
            "Platform": "TikTok",
            "Video Category": "Jokes/Memes",
            "Frequency": "Evening",
            "Watch Time": "Afternoon",
        },
    ),
]


# =========================================================
# 2. Feature weights
# =========================================================

NUMERIC_WEIGHTS = {
    "Number of Videos Watched": 1.00,
    "Importance Score": 0.90,
    "Time Spent On Video": 0.75,
    "Engagement": 0.70,
    "Video Length": 0.55,
    "Total Time Spent": 0.45,
    "Number of Sessions": 0.40,
    "Scroll Rate": 0.35,
}

CATEGORICAL_WEIGHTS = {
    "Watch Time": 1.00,
    "Video Category": 0.75,
    "Frequency": 0.70,
    "Platform": 0.65,
}


# =========================================================
# 3. Answer mappings
# =========================================================

VIDEO_LENGTH_MAP = {
    "A": 3.0,
    "B": 9.0,
    "C": 15.0,
    "D": 21.0,
    "E": 27.0,
}

TIME_SPENT_ON_VIDEO_MAP = {
    "A": 5.0,
    "B": 11.0,
    "C": 15.0,
    "D": 19.0,
    "E": 25.0,
}

SCROLL_RATE_MAP = {
    "A": 15.0,
    "B": 35.0,
    "C": 50.0,
    "D": 65.0,
    "E": 85.0,
}

ENGAGEMENT_MAP = {
    "A": 1500.0,
    "B": 4000.0,
    "C": 4700.0,
    "D": 6500.0,
    "E": 9000.0,
}

TOTAL_TIME_SPENT_MAP = {
    "Less than 30 minutes": 15.0,
    "30-60 minutes": 45.0,
    "1-2 hours": 90.0,
    "2-3 hours": 150.0,
    "3-4 hours": 210.0,
    "4-6 hours": 300.0,
    "6-8 hours": 420.0,
    "More than 8 hours": 540.0,
}

NUMBER_OF_VIDEOS_WATCHED_MAP = {
    "1–10": 5.0,
    "11–25": 18.0,
    "26–50": 38.0,
    "51–100": 75.0,
    "More than 100": 120.0,
}

NUMBER_OF_SESSIONS_MAP = {
    "1–3": 2.0,
    "4–6": 5.0,
    "7–10": 8.5,
    "11–15": 13.0,
    "More than 15": 18.0,
}

IMPORTANCE_SCORE_MAP = {
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
}


# =========================================================
# 4. Optional detailed engagement block
#    Use either the single engagement question
#    or the detailed interaction block.
# =========================================================

INTERACTION_SCALE_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Very Often": 4,
}

INTERACTION_TO_ENGAGEMENT_VALUE = {
    0: 1500.0,
    1: 3000.0,
    2: 4700.0,
    3: 6500.0,
    4: 9000.0,
}


# =========================================================
# 5. Ranges for numeric normalization
#    Higher match score = better match
# =========================================================

NUMERIC_RANGES = {
    "Total Time Spent": (15.0, 540.0),
    "Number of Sessions": (2.0, 18.0),
    "Video Length": (3.0, 27.0),
    "Time Spent On Video": (5.0, 25.0),
    "Number of Videos Watched": (5.0, 120.0),
    "Scroll Rate": (15.0, 85.0),
    "Engagement": (1500.0, 9000.0),
    "Importance Score": (1.0, 9.0),
}


# =========================================================
# 6. Validation sets
# =========================================================

PLATFORM_ALLOWED = {"TikTok", "Instagram", "YouTube", "Facebook"}
VIDEO_CATEGORY_ALLOWED = {"Jokes/Memes", "Life Hacks", "Gaming", "Vlogs", "Pranks"}
FREQUENCY_ALLOWED = {"Morning", "Afternoon", "Evening", "Night"}


# =========================================================
# 7. Normalization helpers
# =========================================================

def normalize_watch_time_str(time_str: str) -> str:
    """
    Convert HH:MM string into one of:
    Morning / Afternoon / Evening / Night
    Example:
        "14:35" -> "Afternoon"
        "21:10" -> "Night"
    """
    try:
        hour, minute = map(int, time_str.split(":"))
    except Exception as e:
        raise ValueError(f"Invalid Watch Time format: {time_str}. Expected 'HH:MM'.") from e

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid Watch Time value: {time_str}. Hour must be 00-23 and minute 00-59.")

    total_minutes = hour * 60 + minute

    if 5 * 60 <= total_minutes < 12 * 60:
        return "Morning"
    elif 12 * 60 <= total_minutes < 17 * 60:
        return "Afternoon"
    elif 17 * 60 <= total_minutes < 21 * 60:
        return "Evening"
    else:
        return "Night"


def map_answer(mapping: Dict[str, float], answer: Optional[str], field_name: str) -> Optional[float]:
    if answer is None:
        return None
    if answer not in mapping:
        raise ValueError(f"Unexpected answer for {field_name}: {answer}")
    return float(mapping[answer])


def derive_engagement_from_interactions(interactions: Dict[str, str]) -> float:
    """
    interactions example:
    {
        "Like": "Often",
        "Comment": "Rarely",
        "Share": "Sometimes",
        "Save": "Often",
        "Send": "Sometimes",
        "Follow": "Rarely",
    }
    """
    if not interactions:
        raise ValueError("interactions cannot be empty")

    vals = []
    for _, answer in interactions.items():
        if answer not in INTERACTION_SCALE_MAP:
            raise ValueError(f"Unexpected interaction answer: {answer}")
        vals.append(INTERACTION_SCALE_MAP[answer])

    avg_score = sum(vals) / len(vals)
    rounded = int(round(avg_score))
    rounded = max(0, min(4, rounded))
    return INTERACTION_TO_ENGAGEMENT_VALUE[rounded]


def normalize_user_answers(raw_answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected raw_answers example:

    {
        "Platform": "YouTube",
        "Frequency": "Evening",
        "Video Category": "Jokes/Memes",
        "Watch Time": "14:35",

        "Video Length": "D",
        "Total Time Spent": "2-3 hours",
        "Number of Videos Watched": "26–50",
        "Number of Sessions": "7–10",
        "Importance Score": "4",

        # choose ONE engagement input style:
        "Engagement": "B",
        # or:
        # "Interactions": {
        #     "Like": "Often",
        #     "Comment": "Rarely",
        #     "Share": "Sometimes",
        #     "Save": "Often",
        #     "Send": "Sometimes",
        #     "Follow": "Rarely",
        # }

        "Time Spent On Video": "C",
        "Scroll Rate": "B",
    }
    """
    features: Dict[str, Any] = {}

    # -------------------------
    # categorical features
    # -------------------------
    platform = raw_answers.get("Platform")
    if platform is not None:
        if platform not in PLATFORM_ALLOWED:
            raise ValueError(f"Unexpected Platform answer: {platform}")
        features["Platform"] = platform

    frequency = raw_answers.get("Frequency")
    if frequency is not None:
        if frequency not in FREQUENCY_ALLOWED:
            raise ValueError(f"Unexpected Frequency answer: {frequency}")
        features["Frequency"] = frequency

    video_category = raw_answers.get("Video Category")
    if video_category is not None:
        if video_category not in VIDEO_CATEGORY_ALLOWED:
            raise ValueError(f"Unexpected Video Category answer: {video_category}")
        features["Video Category"] = video_category

    watch_time = raw_answers.get("Watch Time")
    if watch_time is not None:
        features["Watch Time"] = normalize_watch_time_str(watch_time)

    # -------------------------
    # numeric features
    # -------------------------
    features["Video Length"] = map_answer(VIDEO_LENGTH_MAP, raw_answers.get("Video Length"), "Video Length")
    features["Time Spent On Video"] = map_answer(
        TIME_SPENT_ON_VIDEO_MAP, raw_answers.get("Time Spent On Video"), "Time Spent On Video"
    )
    features["Scroll Rate"] = map_answer(SCROLL_RATE_MAP, raw_answers.get("Scroll Rate"), "Scroll Rate")
    features["Total Time Spent"] = map_answer(
        TOTAL_TIME_SPENT_MAP, raw_answers.get("Total Time Spent"), "Total Time Spent"
    )
    features["Number of Videos Watched"] = map_answer(
        NUMBER_OF_VIDEOS_WATCHED_MAP, raw_answers.get("Number of Videos Watched"), "Number of Videos Watched"
    )
    features["Number of Sessions"] = map_answer(
        NUMBER_OF_SESSIONS_MAP, raw_answers.get("Number of Sessions"), "Number of Sessions"
    )

    if raw_answers.get("Importance Score") is not None:
        score_str = str(raw_answers["Importance Score"])
        features["Importance Score"] = map_answer(IMPORTANCE_SCORE_MAP, score_str, "Importance Score")

    # engagement: prefer interaction block if given
    interactions = raw_answers.get("Interactions")
    if interactions:
        features["Engagement"] = derive_engagement_from_interactions(interactions)
    else:
        features["Engagement"] = map_answer(ENGAGEMENT_MAP, raw_answers.get("Engagement"), "Engagement")

    return features


# =========================================================
# 8. Scoring helpers
# =========================================================

def numeric_match_score(feature_name: str, user_val: float, proto_val: float) -> float:
    """
    Return a score in [0, 1], where higher means closer match.
    """
    low, high = NUMERIC_RANGES[feature_name]
    feature_range = max(high - low, 1e-6)
    distance = abs(user_val - proto_val) / feature_range
    return max(0.0, 1.0 - distance)


def categorical_match_score(user_val: str, proto_val: str) -> float:
    return 1.0 if user_val == proto_val else 0.0


def score_persona(user_features: Dict[str, Any], prototype: PersonaPrototype) -> Tuple[float, List[Tuple[str, float]]]:
    total_score = 0.0
    feature_scores: List[Tuple[str, float]] = []

    # numeric features
    for feature_name, weight in NUMERIC_WEIGHTS.items():
        user_val = user_features.get(feature_name)
        proto_val = prototype.numeric_means.get(feature_name)
        if user_val is None or proto_val is None:
            continue

        raw_score = numeric_match_score(feature_name, float(user_val), float(proto_val))
        weighted_score = raw_score * weight
        total_score += weighted_score
        feature_scores.append((feature_name, weighted_score))

    # categorical features
    for feature_name, weight in CATEGORICAL_WEIGHTS.items():
        user_val = user_features.get(feature_name)
        proto_val = prototype.categorical_modes.get(feature_name)
        if user_val is None or proto_val is None:
            continue

        raw_score = categorical_match_score(str(user_val), str(proto_val))
        weighted_score = raw_score * weight
        total_score += weighted_score
        feature_scores.append((feature_name, weighted_score))

    feature_scores.sort(key=lambda x: x[1], reverse=True)
    return total_score, feature_scores


# =========================================================
# 9. Interpretation and suggestions
# =========================================================

def generate_interpretation(persona_name: str) -> str:
    if persona_name == "Long-Session Heavy Viewers":
        return (
            "This person is most similar to the long-session heavy viewing type.\n"
            "They are more likely to spend a long total time on social media,\n"
            "watch many videos, and stay with longer content."
        )
    elif persona_name == "High-Engagement Night Habit Users":
        return (
            "This person is most similar to the high-engagement night habit type.\n"
            "They are more likely to use social media at night,\n"
            "feel highly attached to the content, and show stronger engagement."
        )
    else:
        return (
            "This person is most similar to the frequent interruption scrolling type.\n"
            "They are more likely to check social media many times a day,\n"
            "scroll quickly, and experience more fragmented viewing habits."
        )


def generate_prevention_ideas(persona_name: str) -> List[str]:
    if persona_name == "Long-Session Heavy Viewers":
        return [
            "session-length reminders",
            "break prompts",
            "daily watch-limit goals",
        ]
    elif persona_name == "High-Engagement Night Habit Users":
        return [
            "night-time cutoff reminders",
            "bedtime app limits",
            "reduced high-attachment content exposure",
        ]
    else:
        return [
            "check-in batching",
            "focus-mode prompts during work or study",
            "short-session interruption limits",
        ]


# =========================================================
# 10. Main classification
# =========================================================

def classify_persona(raw_answers: Dict[str, Any]) -> Dict[str, Any]:
    user_features = normalize_user_answers(raw_answers)

    all_scores: Dict[str, float] = {}
    all_feature_scores: Dict[str, List[Tuple[str, float]]] = {}

    for prototype in PERSONA_PROTOTYPES:
        score, feature_scores = score_persona(user_features, prototype)
        all_scores[prototype.name] = round(score, 4)
        all_feature_scores[prototype.name] = feature_scores

    best_persona = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_persona]
    top_matching_features = all_feature_scores[best_persona][:6]

    return {
        "Predicted Persona": best_persona,
        "Best Score": best_score,
        "All Scores": all_scores,
        "Top Matching Features": top_matching_features,
        "Interpretation": generate_interpretation(best_persona),
        "Suggested prevention ideas": generate_prevention_ideas(best_persona),
        "Normalized Features": user_features,
    }


# =========================================================
# 11. Pretty print
# =========================================================

def print_result(result: Dict[str, Any]) -> None:
    print("=== Result ===")
    print(f"Predicted Persona: {result['Predicted Persona']}")
    print(f"Best Score: {result['Best Score']}")
    print(f"All Scores: {result['All Scores']}")
    print(f"Top Matching Features: {result['Top Matching Features']}")
    print()

    print("Interpretation:")
    print(result["Interpretation"])
    print()

    print("Suggested prevention ideas:")
    for idea in result["Suggested prevention ideas"]:
        print(f"- {idea}")


# =========================================================
# 12. Example test
# =========================================================

if __name__ == "__main__":
    sample_answers = {
        "Platform": "YouTube",
        "Frequency": "Evening",
        "Video Category": "Jokes/Memes",
        "Watch Time": "14:35",
        "Video Length": "D",
        "Total Time Spent": "2-3 hours",
        "Number of Videos Watched": "26–50",
        "Number of Sessions": "7–10",
        "Importance Score": "4",
        "Engagement": "B",
        "Time Spent On Video": "C",
        "Scroll Rate": "B",
    }

    result = classify_persona(sample_answers)
    print_result(result)