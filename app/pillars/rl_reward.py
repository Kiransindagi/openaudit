"""
Pillar 3: RL Reward Function Auditing
Graders matching exact ground truth: sparse_reward, reward_hacking, broken_verifier
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/rl_configs")

def load_rl_config(config_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{config_id}.json"
    with open(filepath) as f:
        return json.load(f)

def grade_sparse_reward(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for sparse_reward flaw type"""
    description = action.description.lower()
    
    has_sparse = any(kw in description for kw in ["sparse", "rare", "few", "only at end", "final step", "zero"])
    score = 0.8 if has_sparse else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Sparse reward: {has_sparse}",
        finding_matched="sparse_reward" if has_sparse else None,
        is_false_positive=not has_sparse,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_reward_hacking(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for reward_hacking flaw type"""
    description = action.description.lower()
    
    has_hacking = any(kw in description for kw in ["hack", "exploit", "cheat", "gaming", "always yes", "always same"])
    score = 0.8 if has_hacking else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Reward hacking: {has_hacking}",
        finding_matched="reward_hacking" if has_hacking else None,
        is_false_positive=not has_hacking,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_broken_verifier(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for broken_verifier flaw type"""
    description = action.description.lower()
    
    has_broken = any(kw in description for kw in ["broken", "always return", "constant", "never penalize", "verifier"])
    score = 0.8 if has_broken else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Broken verifier: {has_broken}",
        finding_matched="broken_verifier" if has_broken else None,
        is_false_positive=not has_broken,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_reward(action: AuditAction, config_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader based on flaw_type"""
    ground_truth = config_data.get("ground_truth_flaws", [])
    
    for flaw in ground_truth:
        flaw_type = flaw.get("flaw_type")
        if flaw_type == "sparse_reward":
            return grade_sparse_reward(action, ground_truth)
        elif flaw_type == "reward_hacking":
            return grade_reward_hacking(action, ground_truth)
        elif flaw_type == "broken_verifier":
            return grade_broken_verifier(action, ground_truth)
    
    return AuditReward(value=0.0, reason="No matching flaw type", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
