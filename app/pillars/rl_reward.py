"""
Pillar 3: RL Reward Auditing - Corrected for 'type' field
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
    """Grader that looks for 'type' field in ground truth"""
    description = action.description.lower()
    ground_truth = config_data.get("ground_truth_flaws", [])
    
    # Find the flaw with type='sparse_reward'
    target_flaw = None
    for flaw in ground_truth:
        if flaw.get("type") == "sparse_reward":
            target_flaw = flaw
            break
    
    if target_flaw is None:
        return AuditReward(
            value=0.0,
            reason=f"No sparse_reward flaw. Found types: {[f.get('type') for f in ground_truth]}",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    # Check for sparse-related keywords
    if any(kw in description for kw in ["sparse", "rare", "only at end", "final step"]):
        return AuditReward(
            value=0.8,
            reason="Correctly identified sparse reward",
            finding_matched="sparse_reward",
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.8
        )
    else:
        return AuditReward(
            value=0.2,
            reason="Partial credit - sparse reward exists",
            finding_matched=None,
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.2
        )
