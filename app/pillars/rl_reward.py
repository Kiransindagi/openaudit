"""
Pillar 3: RL Reward Auditing - Fixed routing
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/rl_configs")

def load_rl_config(config_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{config_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_sparse_reward(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    if any(kw in description for kw in ["sparse", "rare", "only at end"]):
        return AuditReward(value=0.8, reason="Sparse reward detected", finding_matched="sparse_reward", is_false_positive=False, penalty_applied=0.0, cumulative_score=0.8)
    else:
        return AuditReward(value=0.2, reason="Partial: sparse reward exists", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.2)

def grade_reward_hacking(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    # Find the reward_hacking flaw
    hacking_flaw = None
    for flaw in ground_truth:
        if flaw.get("type") == "reward_hacking" or flaw.get("flaw_type") == "reward_hacking":
            hacking_flaw = flaw
            break
    if not hacking_flaw:
        return AuditReward(value=0.0, reason="No reward hacking flaw", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    trigger = hacking_flaw.get("trigger", "yes").lower()
    if trigger in description or any(kw in description for kw in ["hack", "exploit", "cheat"]):
        return AuditReward(value=0.8, reason="Reward hacking detected", finding_matched="reward_hacking", is_false_positive=False, penalty_applied=0.0, cumulative_score=0.8)
    else:
        return AuditReward(value=0.2, reason="Partial: reward hacking exists", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.2)

def grade_broken_verifier(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    if any(kw in description for kw in ["broken", "always return", "constant"]):
        return AuditReward(value=0.8, reason="Broken verifier detected", finding_matched="broken_verifier", is_false_positive=False, penalty_applied=0.0, cumulative_score=0.8)
    else:
        return AuditReward(value=0.2, reason="Partial: broken verifier exists", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.2)

def grade_reward(action: AuditAction, config_data: Dict[str, Any]) -> AuditReward:
    ground_truth = config_data.get("ground_truth_flaws", [])
    # Iterate over flaws to find the correct one
    for flaw in ground_truth:
        flaw_type = flaw.get("type") or flaw.get("flaw_type", "")
        if flaw_type == "sparse_reward":
            return grade_sparse_reward(action, ground_truth)
        elif flaw_type == "reward_hacking":
            return grade_reward_hacking(action, ground_truth)
        elif flaw_type == "broken_verifier":
            return grade_broken_verifier(action, ground_truth)
    # If no matching flaw
    return AuditReward(value=0.0, reason=f"No matching flaw. Types: {[f.get('type') or f.get('flaw_type') for f in ground_truth]}", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
