"""
Pillar 4: Tool Tester - Updated for actual ground truth types
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/tools")

def load_tool(tool_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{tool_id}.json"
    with open(filepath) as f:
        return json.load(f)

def grade_code_quality(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for code_quality issues"""
    description = action.description.lower()
    
    # Find the code_quality flaw
    target_flaw = None
    for flaw in ground_truth:
        if flaw.get("type") == "code_quality":
            target_flaw = flaw
            break
    
    if not target_flaw:
        return AuditReward(value=0.0, reason="No code_quality flaw", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    expected_issues = set(target_flaw.get("issues", []))
    total_issues = len(expected_issues)
    
    # Check for keywords in description
    found_issues = []
    if any(kw in description for kw in ["docstring", "documentation"]):
        found_issues.append("no_docstring")
    if any(kw in description for kw in ["type hint", "type annotation", "typing"]):
        found_issues.append("no_type_hints")
    if any(kw in description for kw in ["return", "returns", "output type"]):
        found_issues.append("no_return_annotation")
    
    correct = len(set(found_issues) & expected_issues)
    score = correct / total_issues if total_issues > 0 else 0.0
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct}/{total_issues} code quality issues",
        finding_matched="code_quality" if score > 0 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_silent_failure(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for silent_failure"""
    description = action.description.lower()
    
    has_silent = any(kw in description for kw in ["silent", "exception", "swallow", "return none", "error ignored"])
    score = 0.8 if has_silent else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Silent failure detection: {has_silent}",
        finding_matched="silent_failure" if has_silent else None,
        is_false_positive=not has_silent,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_adversarial_chain(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for adversarial_chain"""
    description = action.description.lower()
    
    has_dangerous = any(kw in description for kw in ["dangerous", "exec", "eval", "chain", "tool c", "adversarial"])
    score = 0.8 if has_dangerous else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Adversarial chain detection: {has_dangerous}",
        finding_matched="adversarial_chain" if has_dangerous else None,
        is_false_positive=not has_dangerous,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader based on flaw type"""
    ground_truth = tool_data.get("ground_truth_flaws", [])
    
    for flaw in ground_truth:
        flaw_type = flaw.get("type")
        if flaw_type == "code_quality":
            return grade_code_quality(action, ground_truth)
        elif flaw_type == "silent_failure":
            return grade_silent_failure(action, ground_truth)
        elif flaw_type == "adversarial_chain":
            return grade_adversarial_chain(action, ground_truth)
    
    return AuditReward(value=0.0, reason="Unknown flaw type", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
