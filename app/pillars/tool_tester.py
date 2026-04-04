"""
Pillar 4: smolagents Tool Tester
Deterministic graders for tool reliability issues
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/tools")

def load_tool(tool_id: str) -> Dict[str, Any]:
    """Load tool by ID (tool_0 through tool_9)"""
    filepath = DATA_DIR / f"{tool_id}.json"
    with open(filepath) as f:
        return json.load(f)

def grade_static_analysis(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Easy grader: Detect missing code quality elements"""
    description = action.description.lower()
    
    static_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "static_analysis":
            static_flaw = flaw
            break
    
    if not static_flaw:
        return AuditReward(
            value=0.0,
            reason="No static analysis issues",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_issues = set(static_flaw.get("missing_elements", []))
    total_issues = len(expected_issues)
    
    # Check for issue keywords
    agent_issues = set()
    issue_keywords = {
        "docstring": ["docstring", "documentation", "docs"],
        "type_hints": ["type hint", "type annotation", "typing"],
        "return_annotation": ["return", "returns", "output type"]
    }
    
    for issue, keywords in issue_keywords.items():
        if any(kw in description for kw in keywords):
            agent_issues.add(issue)
    
    correct_matches = len(agent_issues & expected_issues)
    score = correct_matches / total_issues if total_issues > 0 else 1.0
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct_matches}/{total_issues} missing elements",
        finding_matched=f"static_analysis:{list(agent_issues)}" if correct_matches > 0 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_silent_failure(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Medium grader: Detect silent exception handling"""
    description = action.description.lower()
    
    silent_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "silent_failure":
            silent_flaw = flaw
            break
    
    if not silent_flaw:
        return AuditReward(
            value=0.0,
            reason="No silent failure issues",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    # Check for issue detection
    has_exception = any(kw in description for kw in ["exception", "error", "try", "except", "swallow"])
    has_none_return = any(kw in description for kw in ["returns none", "none", "null", "silent"])
    
    score = 0.0
    if has_exception:
        score += 0.5
    if has_none_return:
        score += 0.5
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Silent failure: exception_hiding={has_exception}, returns_none={has_none_return}",
        finding_matched="silent_failure" if score >= 0.5 else None,
        is_false_positive=score < 0.3,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_adversarial_chain(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Hard grader: Identify dangerous tool chain"""
    description = action.description.lower()
    
    chain_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "adversarial_chain":
            chain_flaw = flaw
            break
    
    if not chain_flaw:
        return AuditReward(
            value=0.0,
            reason="No adversarial chain issues",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_dangerous = chain_flaw.get("dangerous_tool", "").lower()
    
    # Check for identification
    has_dangerous = expected_dangerous in description if expected_dangerous else False
    has_reason = any(kw in description for kw in ["exec", "eval", "dangerous", "security", "injection", "malicious"])
    
    score = 0.0
    if has_dangerous:
        score += 0.5
    if has_reason:
        score += 0.5
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Adversarial chain: identifies={has_dangerous}, explains={has_reason}",
        finding_matched="adversarial_chain" if score >= 0.6 else None,
        is_false_positive=score < 0.3,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader"""
    ground_truth = tool_data.get("ground_truth_flaws", [])
    flaw_types = [f["flaw_type"] for f in ground_truth]
    
    if "static_analysis" in flaw_types:
        return grade_static_analysis(action, ground_truth)
    elif "silent_failure" in flaw_types:
        return grade_silent_failure(action, ground_truth)
    elif "adversarial_chain" in flaw_types:
        return grade_adversarial_chain(action, ground_truth)
    else:
        return AuditReward(
            value=0.0,
            reason="Unknown flaw type",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )

def test_graders():
    """Test tool tester graders"""
    print("Testing Tool Tester Graders...")
    print("Note: Create synthetic tool files first!")
    print("✅ Placeholder - implement with actual data")

if __name__ == "__main__":
    test_graders()
