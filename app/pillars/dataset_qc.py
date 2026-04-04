"""
Pillar 2: Dataset Quality Control Auditing
Graders matching exact ground truth flaw types: null_values, duplicates, test_leakage
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/datasets")

def load_dataset(dataset_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{dataset_id}.json"
    with open(filepath) as f:
        return json.load(f)

def grade_null_values(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for null_values flaw type"""
    description = action.description.lower()
    
    # Find the null_values flaw in ground truth
    null_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "null_values":
            null_flaw = flaw
            break
    
    if not null_flaw:
        return AuditReward(value=0.0, reason="No null_values flaw", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    expected_columns = set(null_flaw.get("columns", []))
    total_nulls = null_flaw.get("total_nulls", 0)
    
    # Check for keywords
    has_nulls = any(kw in description for kw in ["null", "missing", "empty"])
    score = 0.6 if has_nulls else 0.0
    
    # Bonus for mentioning columns
    columns_found = sum(1 for col in expected_columns if col.lower() in description)
    if expected_columns:
        score += (columns_found / len(expected_columns)) * 0.4
    
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason=f"Null values: found {columns_found}/{len(expected_columns)} columns",
        finding_matched="null_values" if score > 0.5 else None,
        is_false_positive=score < 0.3,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_duplicates(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for duplicates flaw type"""
    description = action.description.lower()
    
    has_duplicates = any(kw in description for kw in ["duplicate", "duplicated", "identical", "same rows"])
    score = 0.8 if has_duplicates else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Duplicate detection: {has_duplicates}",
        finding_matched="duplicates" if has_duplicates else None,
        is_false_positive=not has_duplicates,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_test_leakage(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Grader for test_leakage flaw type"""
    description = action.description.lower()
    
    has_leakage = any(kw in description for kw in ["leak", "leakage", "train", "test", "overlap"])
    score = 0.8 if has_leakage else 0.2
    
    return AuditReward(
        value=score,
        reason=f"Test leakage detection: {has_leakage}",
        finding_matched="test_leakage" if has_leakage else None,
        is_false_positive=not has_leakage,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_dataset(action: AuditAction, dataset_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader based on flaw_type"""
    ground_truth = dataset_data.get("ground_truth_flaws", [])
    
    for flaw in ground_truth:
        flaw_type = flaw.get("flaw_type")
        if flaw_type == "null_values":
            return grade_null_values(action, ground_truth)
        elif flaw_type == "duplicates":
            return grade_duplicates(action, ground_truth)
        elif flaw_type == "test_leakage":
            return grade_test_leakage(action, ground_truth)
    
    return AuditReward(value=0.0, reason="No matching flaw type", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
