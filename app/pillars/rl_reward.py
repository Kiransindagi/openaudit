"""
Pillar 3: RL Reward Auditing - All graders give base 0.2
"""
import json
import re

def _clamp_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1 (never 0.0 or 1.0)."""
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value
from pathlib import Path

def _clamp_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1 (never 0.0 or 1.0)."""
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value
from typing import Dict, List, Any

def _clamp_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1 (never 0.0 or 1.0)."""
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value
from app.models import AuditAction, AuditReward

def _clamp_reward(value: float) -> float:
    """Ensure reward is strictly between 0 and 1 (never 0.0 or 1.0)."""
    if value <= 0.0:
        return 0.001
    if value >= 1.0:
        return 0.999
    return value

DATA_DIR = Path("data/rl_configs")

def load_rl_config(config_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{config_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_sparse_reward(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2
    if any(kw in description for kw in ["sparse", "rare", "only at end"]):
        score += 0.6
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Sparse reward detection",
        finding_matched="sparse_reward" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_reward_hacking(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2
    if any(kw in description for kw in ["hack", "exploit", "cheat", "always", "same"]):
        score += 0.6
    # Also check for trigger word if present in ground truth
    for flaw in ground_truth:
        if flaw.get("type") == "reward_hacking" or flaw.get("flaw_type") == "reward_hacking":
            trigger = flaw.get("trigger", "").lower()
            if trigger and trigger in description:
                score = max(score, 0.8)
                break
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Reward hacking detection",
        finding_matched="reward_hacking" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_broken_verifier(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2
    if any(kw in description for kw in ["broken", "always return", "constant", "never penalize"]):
        score += 0.6
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Broken verifier detection",
        finding_matched="broken_verifier" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_reward(action: AuditAction, config_data: Dict[str, Any]) -> AuditReward:
    ground_truth = config_data.get("ground_truth_flaws", [])
    for flaw in ground_truth:
        flaw_type = flaw.get("type") or flaw.get("flaw_type", "")
        if flaw_type == "sparse_reward":
            return grade_sparse_reward(action, ground_truth)
        elif flaw_type == "reward_hacking":
            return grade_reward_hacking(action, ground_truth)
        elif flaw_type == "broken_verifier":
            return grade_broken_verifier(action, ground_truth)
    # Fallback
    return AuditReward(
        value=_clamp_reward(0.21),
        reason="Partial credit – action recognized",
        finding_matched=None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=0.2
    )


