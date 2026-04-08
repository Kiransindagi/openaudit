"""
Pillar 2: Dataset Quality Control - Complete with all three difficulty graders
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

DATA_DIR = Path("data/datasets")

def load_dataset(dataset_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{dataset_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_null_values(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2  # base reward for any valid action
    if any(kw in description for kw in ["null", "missing", "empty"]):
        score += 0.6
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Null detection",
        finding_matched="null_values" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_duplicates(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2
    if any(kw in description for kw in ["duplicate", "identical", "same rows"]):
        score += 0.6
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Duplicate detection",
        finding_matched="duplicates" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_test_leakage(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    score = 0.2
    if any(kw in description for kw in ["leak", "leakage", "train", "test", "overlap"]):
        score += 0.6
    return AuditReward(
        value=_clamp_reward(round(min(0.99), max(0.01, score)), 3),
        reason="Test leakage detection",
        finding_matched="test_leakage" if score > 0.5 else None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_dataset(action: AuditAction, dataset_data: Dict[str, Any]) -> AuditReward:
    ground_truth = dataset_data.get("ground_truth_flaws", [])
    for flaw in ground_truth:
        flaw_type = flaw.get("type") or flaw.get("flaw_type", "")
        if flaw_type == "null_values":
            return grade_null_values(action, ground_truth)
        elif flaw_type == "duplicates":
            return grade_duplicates(action, ground_truth)
        elif flaw_type == "test_leakage":
            return grade_test_leakage(action, ground_truth)
    # Fallback for any other case
    return AuditReward(
        value=_clamp_reward(0.21),
        reason="Partial credit – action recognized",
        finding_matched=None,
        is_false_positive=False,
        penalty_applied=0.0,
        cumulative_score=0.2
    )


