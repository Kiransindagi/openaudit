"""
Pillar 4: Tool Tester - Fixed for code_quality flaw type
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

def grade_tool(action: AuditAction, tool_data: Dict[str, Any]) -> AuditReward:
    """Grader for tool quality issues"""
    description = action.description.lower()
    ground_truth = tool_data.get("ground_truth_flaws", [])
    
    # Find code_quality flaw
    target_flaw = None
    for flaw in ground_truth:
        if flaw.get("type") == "code_quality":
            target_flaw = flaw
            break
    
    if target_flaw is None:
        return AuditReward(
            value=0.0,
            reason=f"No code_quality flaw found. Types: {[f.get('type') for f in ground_truth]}",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_issues = set(target_flaw.get("issues", []))
    
    # Check for keywords in description
    found_issues = []
    if any(kw in description for kw in ["docstring", "documentation"]):
        found_issues.append("no_docstring")
    if any(kw in description for kw in ["type hint", "type annotation", "typing"]):
        found_issues.append("no_type_hints")
    if any(kw in description for kw in ["return", "returns", "output type"]):
        found_issues.append("no_return_annotation")
    
    score = len(found_issues) / len(expected_issues) if expected_issues else 0.8
    
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason=f"Found {len(found_issues)}/{len(expected_issues)} code quality issues",
        finding_matched="code_quality",
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )
