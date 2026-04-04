"""
Pillar 4: Tool Tester - Always returns base 0.2 for any valid action
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
    # Always give 0.2 for any action with pillar = tool_tester
    if action.pillar == "tool_tester":
        return AuditReward(
            value=0.3,
            reason="Partial credit – action recognized",
            finding_matched=None,
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.3
        )
    return AuditReward(
        value=0.0,
        reason="Invalid pillar",
        finding_matched=None,
        is_false_positive=True,
        penalty_applied=0.0,
        cumulative_score=0.0
    )

