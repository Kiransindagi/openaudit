"""
Pillar 4: Tool Tester - Grades findings against ground truth flaw types
"""
import json
from pathlib import Path
from typing import Dict, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/tools")

def load_tool(tool_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{tool_id}.json"
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    if action.pillar != "tool_tester":
        return AuditReward(
            value=0.0, reason="Invalid pillar",
            finding_matched=None, is_false_positive=True,
            penalty_applied=0.0, cumulative_score=0.0
        )

    ground_truth = tool_data.get("ground_truth_flaws", [])
    description = action.description.lower()
    finding_type = (action.finding_type or "").lower()

    flaw_types = [f.get("type") for f in ground_truth]
    primary_flaw = flaw_types[0] if flaw_types else None

    # --- code_quality ---
    if primary_flaw == "code_quality":
        expected_issues = set()
        for flaw in ground_truth:
            if flaw.get("type") == "code_quality":
                for issue in flaw.get("issues", []):
                    expected_issues.add(issue)

        issue_keywords = {
            "no_docstring":      ["docstring", "no doc", "missing doc", "undocumented"],
            "no_type_hints":     ["type hint", "type annotation", "no type", "missing type"],
            "no_return":         ["return", "no return", "missing return"],
            "unused_variable":   ["unused", "dead code", "unreferenced"],
            "no_error_handling": ["error handling", "exception", "try", "catch"],
            "magic_number":      ["magic number", "hardcoded", "literal"],
        }

        matched = set()
        for issue, keywords in issue_keywords.items():
            if issue in expected_issues and any(kw in description for kw in keywords):
                matched.add(issue)

        type_bonus = 0.1 if "code_quality" in finding_type else 0.0
        score = (len(matched) / len(expected_issues)) if expected_issues else 0.0
        score = min(1.0, score + type_bonus)

        return AuditReward(
            value=round(min(0.99, max(0.21, score)), 3),
            reason=f"Matched {len(matched)}/{len(expected_issues)} code quality issues",
            finding_matched=f"code_quality:{list(matched)}" if matched else None,
            is_false_positive=len(matched) == 0,
            penalty_applied=0.0, cumulative_score=round(score, 3)
        )

    # --- silent_failure ---
    elif primary_flaw == "silent_failure":
        keywords = ["silent", "swallow", "bare except", "except:", "return none",
                    "silent failure", "exception ignored", "no logging", "suppressed"]
        type_keywords = ["silent_failure", "silent failure", "error_suppression"]

        desc_match = any(kw in description for kw in keywords)
        type_match = any(kw in finding_type for kw in type_keywords)
        score = 0.0
        if desc_match: score += 0.6
        if type_match: score += 0.4

        return AuditReward(
            value=round(max(0.21, min(0.99, score)), 3),
            reason="Silent failure detection",
            finding_matched="silent_failure" if score >= 0.6 else None,
            is_false_positive=score == 0.0,
            penalty_applied=0.0, cumulative_score=round(score, 3)
        )

    # --- adversarial_chain ---
    elif primary_flaw == "adversarial_chain":
        keywords = ["exec", "arbitrary code", "code injection", "unsafe", "adversarial",
                    "remote code", "rce", "injection", "security", "dangerous"]
        type_keywords = ["adversarial_chain", "adversarial", "code_injection", "security"]

        desc_match = any(kw in description for kw in keywords)
        type_match = any(kw in finding_type for kw in type_keywords)
        score = 0.0
        if desc_match: score += 0.6
        if type_match: score += 0.4

        return AuditReward(
            value=round(max(0.21, min(0.99, score)), 3),
            reason="Adversarial chain detection",
            finding_matched="adversarial_chain" if score >= 0.6 else None,
            is_false_positive=score == 0.0,
            penalty_applied=0.0, cumulative_score=round(score, 3)
        )

    # --- unknown flaw type ---
    else:
        return AuditReward(
            value=0.21, reason=f"Unrecognized flaw type: {primary_flaw}",
            finding_matched=None, is_false_positive=False,
            penalty_applied=0.0, cumulative_score=0.2
        )

