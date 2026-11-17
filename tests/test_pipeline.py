from __future__ import annotations

from app import run_pipeline
from data.sample_inputs import get_sample_questionnaire


def test_pipeline_runs_end_to_end() -> None:
    output = run_pipeline(get_sample_questionnaire())

    assert "persona_result" in output
    assert "addiction_result" in output
    assert "profile" in output
    assert "retrieved" in output
    assert "final_response" in output

    assert output["profile"]["persona"]
    assert 1 <= int(output["profile"]["addiction_level"]) <= 10
    assert len(output["retrieved"]) >= 3


def test_final_response_sections_present() -> None:
    text = run_pipeline(get_sample_questionnaire())["final_response"]
    assert "1. Profile Summary" in text
    assert "3. Top 3 Recommended Actions" in text
    assert "5. Longer-Term Prevention Plan" in text
