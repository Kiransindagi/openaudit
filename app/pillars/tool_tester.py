"""
Pillar 4: Tool Tester - Increased base reward to 0.3
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/tools")

def load_tool(tool_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{tool_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_code_quality(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.3  # Base reward
    if any(kw in description for kw in ["docstring", "documentation", "type hint", "return annotation"]):
        score = 0.8
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason="Code quality assessment",
        finding_matched="code_quality" if score >= 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_silent_failure(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.3  # Base reward
    keywords = ["exception", "error", "swallow", "try", "except", "return none", "silent"]
    if any(kw in description for kw in keywords):
        score = 0.8
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason="Silent failure detection",
        finding_matched="silent_failure" if score >= 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_adversarial_chain(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.3  # Base reward
    keywords = ["exec", "eval", "dangerous", "injection", "execute", "arbitrary code"]
    if any(kw in description for kw in keywords):
        score = 0.8
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason="Adversarial chain detection",
        finding_matched="adversarial_chain" if score >= 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    ground_truth = tool_data.get("ground_truth_flaws", [])
    for flaw in ground_truth:
        flaw_type = flaw.get("type") or flaw.get("flaw_type", "")
        if flaw_type == "code_quality":
            return grade_code_quality(action, ground_truth)
        elif flaw_type == "silent_failure":
            return grade_silent_failure(action, ground_truth)
        elif flaw_type == "adversarial_chain":
            return grade_adversarial_chain(action, ground_truth)
    # Fallback - any valid action gets 0.3
    return AuditReward(
        value=0.3,
        reason="Partial credit – action recognized",
        finding_matched=None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=0.3
    )
