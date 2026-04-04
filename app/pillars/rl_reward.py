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
    """Route to appropriate grader based on flaw_type"""
    ground_truth = config_data.get("ground_truth_flaws", [])
    
    # Determine which grader to use based on the first flaw type (or iterate)
    for flaw in ground_truth:
        flaw_type = flaw.get("type") or flaw.get("flaw_type", "")
        if flaw_type == "sparse_reward":
            return grade_sparse_reward(action, ground_truth)
        elif flaw_type == "reward_hacking":
            return grade_reward_hacking(action, ground_truth)
        elif flaw_type == "broken_verifier":
            return grade_broken_verifier(action, ground_truth)
    
    # If no matching flaw found
    return AuditReward(
        value=0.0,
        reason=f"No matching flaw type. Found types: {[f.get('type') or f.get('flaw_type') for f in ground_truth]}",
        finding_matched=None,
        is_false_positive=True,
        penalty_applied=0.0,
        cumulative_score=0.0
    )
