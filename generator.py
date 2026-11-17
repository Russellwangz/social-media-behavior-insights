from __future__ import annotations

from models.schema import RetrievedRecommendation, UserProfile


def _format_action_line(idx: int, rec: RetrievedRecommendation) -> str:
    reason = ", ".join(rec.reasons[:2]) if rec.reasons else "general fit"
    return f"{idx}. {rec.item.title}: {rec.item.content} (fit: {reason})"


def generate_recommendation_response(
    profile: UserProfile,
    retrieved_items: list[RetrievedRecommendation],
) -> str:
    """Generate a local template-based personalized intervention response."""
    top_three = retrieved_items[:3]

    summary = (
        f"Persona: {profile.persona}. "
        f"Addiction level: {profile.addiction_level}/10 ({profile.severity}). "
        f"Key patterns: {', '.join(profile.top_matching_features[:4]) or 'n/a'}."
    )

    if profile.risk_flags:
        impact = (
            "Current usage may affect productivity through "
            + ", ".join(profile.risk_flags)
            + "."
        )
    else:
        impact = "Current pattern shows manageable risk but still benefits from structure and limits."

    actions = [_format_action_line(i, rec) for i, rec in enumerate(top_three, start=1)]

    easy_step = (
        top_three[0].item.content
        if top_three
        else "Set a 20-minute timer for your next social media session and stop when it rings."
    )

    long_term = (
        profile.prevention_ideas[0]
        if profile.prevention_ideas
        else "review usage weekly and adjust limits based on study/work outcomes"
    )

    lines = [
        "1. Profile Summary",
        summary,
        "",
        "2. Why This Pattern May Affect You",
        impact,
        "",
        "3. Top 3 Recommended Actions",
        *actions,
        "",
        "4. Start-Today Action",
        easy_step,
        "",
        "5. Longer-Term Prevention Plan",
        f"For the next 3-4 weeks, keep using: {long_term}.",
    ]
    return "\n".join(lines)
