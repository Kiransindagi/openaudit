"""
Pillar 4: Tool Tester
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
    ground_truth = tool_data.get("ground_truth_flaws", [])
    description = action.description.lower()
    finding_type = (action.finding_type or "").lower()
    primary_flaw = ground_truth[0].get("type") if ground_truth else None

    if primary_flaw == "code_quality":
        expected = set()
        for flaw in ground_truth:
            if flaw.get("type") == "code_quality":
                expected.update(flaw.get("issues", []))
        keywords = {
            "no_docstring":  ["docstring", "no doc", "missing doc", "undocumented"],
            "no_type_hints": ["type hint", "type annotation", "no type", "missing type"],
        }
        matched = {i for i, kws in keywords.items() if i in expected and any(kw in description for kw in kws)}
        bonus = 0.1 if "code_quality" in finding_type else 0.0
        score = round(min(0.99, max(0.21, (len(matched)/max(1,len(expected))) + bonus)), 3)
        return AuditReward(value=score, reason=f"Code quality: {matched}", finding_matched="code_quality" if matched else None, is_false_positive=not matched, penalty_applied=0.01, cumulative_score=score)

    elif primary_flaw == "silent_failure":
        score = 0.21
        if any(kw in description for kw in ["silent", "swallow", "bare except", "return none", "exception ignored"]):
            score += 0.78
        if "silent" in finding_type:
            score += 0.1
        score = round(min(0.99, score), 3)
        return AuditReward(value=score, reason="Silent failure", finding_matched="silent_failure" if score > 0.5 else None, is_false_positive=False, penalty_applied=0.01, cumulative_score=score)

    elif primary_flaw == "adversarial_chain":
        score = 0.21
        if any(kw in description for kw in ["exec", "arbitrary code", "injection", "unsafe", "security", "rce"]):
            score += 0.78
        if "adversarial" in finding_type or "injection" in finding_type:
            score += 0.1
        score = round(min(0.99, score), 3)
        return AuditReward(value=score, reason="Adversarial chain", finding_matched="adversarial_chain" if score > 0.5 else None, is_false_positive=False, penalty_applied=0.01, cumulative_score=score)

    return AuditReward(value=0.21, reason="Unrecognized flaw", finding_matched=None, is_false_positive=False, penalty_applied=0.01, cumulative_score=0.21)

