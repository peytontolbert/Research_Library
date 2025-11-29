from models.mirrormind.coordinator import ReviewAgent


def test_review_agent_outputs_paper_spec_fields():
    repo_plans = [
        {
            "repo_id": "r1",
            "actions": ["Apply patch A", "Rollback patch A"],
            "evidence": ["changed files", "missing tests"],
            "dependencies": ["repo:r1"],
        }
    ]
    paper_plans = [
        {
            "paper_id": "p1",
            "actions": ["TODO: clarify method"],
            "evidence": ["uncertain derivation"],
            "dependencies": {"paper": "p1"},
        }
    ]

    agent = ReviewAgent()
    out = agent.synthesize(repo_plans, paper_plans)

    # Core fields from paper spec section 5.2
    assert "global_plan" in out and out["global_plan"]
    assert "evidence_map" in out and out["evidence_map"]
    assert "evidence_summary" in out
    assert "dependencies" in out
    assert "consistency_report" in out
    assert "issues" in out
    assert "uncertainty_flags" in out

    # Dependencies aggregated per-plan.
    assert "plan_0" in out["dependencies"]
    assert "plan_1" in out["dependencies"]

    # We expect some issues / uncertainty based on the crafted actions/evidence.
    assert any("missing" in issue for issue in out["issues"])
    assert any("plan_contains_todo_or_uncertain_cues" == flag for flag in out["uncertainty_flags"])


