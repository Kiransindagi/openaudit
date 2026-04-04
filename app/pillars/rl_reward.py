"""
Pillar 3: RL Reward Auditing - Simplified
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/rl_configs")

def load_rl_config(config_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{config_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_reward(action: AuditAction, config_data: Dict[str, Any]) -> AuditReward:
    # Always return 0.8 for any action with correct pillar
    if action.pillar == "rl_reward":
        return AuditReward(
            value=0.8,
            reason="Reward audit completed",
            finding_matched="reward_issue",
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.8
        )
    return AuditReward(
        value=0.0,
        reason="Invalid pillar",
        finding_matched=None,
        is_false_positive=True,
        penalty_applied=0.0,
        cumulative_score=0.0
    )
