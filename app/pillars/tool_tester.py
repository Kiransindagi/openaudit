"""
Pillar 4: Tool Tester - Grades code quality findings against ground truth
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/tools")

def load_tool(tool_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{tool_id}.json"
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    if action.pillar != "tool_tester":
        return AuditReward(
            value=0.0,
            reason="Invalid pillar",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )

    ground_truth = tool_data.get("ground_truth_flaws", [])
    description = action.description.lower()
    finding_type = action.finding_type.lower() if action.finding_type else ""

    # Collect all expected issues from ground truth
    expected_issues = set()
    for flaw in ground_truth:
        if flaw.get("type") == "code_quality":
            for issue in flaw.get("issues", []):
                expected_issues.add(issue)

    if not expected_issues:
        return AuditReward(
            value=0.2,
            reason="No code quality flaws in ground truth",
            finding_matched=None,
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.2
        )

    # Keyword mapping for each known issue type
    issue_keywords = {
        "no_docstring":    ["docstring", "no doc", "missing doc", "undocumented"],
        "no_type_hints":   ["type hint", "type annotation", "no type", "missing type"],
        "no_return":       ["return", "no return", "missing return"],
        "unused_variable": ["unused", "dead code", "unreferenced"],
        "no_error_handling": ["error handling", "exception", "try", "catch"],
        "magic_number":    ["magic number", "hardcoded", "literal"],
    }

    matched_issues = set()
    for issue, keywords in issue_keywords.items():
        if issue in expected_issues:
            if any(kw in description for kw in keywords):
                matched_issues.add(issue)

    # Also give credit if finding_type is code_quality
    type_bonus = 0.1 if "code_quality" in finding_type else 0.0

    correct = len(matched_issues)
    total = len(expected_issues)
    score = (correct / total) if total > 0 else 0.0
    score = min(1.0, score + type_bonus)

    false_positive = correct == 0 and type_bonus == 0.0

    return AuditReward(
        value=round(max(0.2, score), 3),
        reason=f"Matched {correct}/{total} code quality issues: {list(matched_issues)}",
        finding_matched=f"code_quality:{list(matched_issues)}" if correct > 0 else None,
        is_false_positive=false_positive,
        penalty_applied=0.0,
        cumulative_score=round(score, 3)
    )