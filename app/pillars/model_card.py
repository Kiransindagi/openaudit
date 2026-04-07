"""
Pillar 1: Model Card Auditing
Deterministic graders for synthetic data with ground truth flaws
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/model_cards")

def load_card(card_id: str) -> Dict[str, Any]:
    """Load model card by ID (card_0 through card_9)"""
    filepath = DATA_DIR / f"{card_id}.json"
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return json.load(f)

def grade_missing_fields(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Easy grader: Field completeness"""
    description = action.description.lower()
    missing_fields: Set[str] = set()
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "missing_field":
            missing_fields.update(flaw.get("fields", []))
    
    if not missing_fields:
        return AuditReward(value=1.0, reason="No missing fields required", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=1.0)
    
    agent_fields: Set[str] = set()
    field_keywords = {
        "license": ["license", "licence"],
        "eval_results": ["eval", "evaluation", "eval_results", "benchmark"],
        "co2_emitted": ["co2", "carbon", "emission", "environmental"],
        "base_model": ["base model", "parent model"],
        "training_data": ["training data", "dataset"]
    }
    
    for field, keywords in field_keywords.items():
        if any(kw in description for kw in keywords):
            agent_fields.add(field)
    
    correct_matches = len(agent_fields & missing_fields)
    total_missing = len(missing_fields)
    score = correct_matches / total_missing if total_missing > 0 else 1.0
    false_positives = len(agent_fields - missing_fields)
    score = max(0.0, score - (false_positives * 0.1))
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct_matches}/{total_missing} missing fields",
        finding_matched=f"missing_field:{list(agent_fields & missing_fields)}" if correct_matches > 0 else None,
        is_false_positive=false_positives > 0,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_license_conflict(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Medium grader: License conflict detection"""
    description = action.description.lower()
    conflict = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "license_conflict":
            conflict = flaw
            break
    
    if not conflict:
        return AuditReward(value=0.0, reason="No license conflict", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    parent_model = conflict.get("parent_model", "").lower()
    checks = {
        "has_license_keyword": "license" in description,
        "has_conflict": any(kw in description for kw in ["conflict", "incompatible", "violation", "gpl"]),
        "has_parent_name": parent_model.split("/")[-1].replace("-", " ") in description or parent_model in description
    }
    score = sum([0.3 for v in checks.values() if v])
    
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason=f"License conflict detection",
        finding_matched="license_conflict" if score >= 0.6 else None,
        is_false_positive=score < 0.3,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_benchmark_fraud(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """Hard grader: Benchmark fraud detection"""
    description = action.description.lower()
    fraud = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "benchmark_fraud":
            fraud = flaw
            break
    
    if not fraud:
        return AuditReward(value=0.0, reason="No benchmark fraud", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    benchmark = fraud.get("benchmark", "").lower()
    claimed = fraud.get("claimed", 0.0)
    actual = fraud.get("actual", 0.0)
    numbers = [float(n) for n in re.findall(r'\d+\.?\d*', description)]
    
    has_benchmark = benchmark in description
    claimed_correct = any(abs(n - claimed) <= 0.5 for n in numbers)
    actual_correct = any(abs(n - actual) <= 0.5 for n in numbers)
    
    score = 0.0
    if has_benchmark:
        score += 0.3
    if claimed_correct:
        score += 0.35
    if actual_correct:
        score += 0.35
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Benchmark fraud detection",
        finding_matched="benchmark_fraud" if score >= 0.6 else None,
        is_false_positive=score < 0.3,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_model_card(action: AuditAction, card_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader"""
    ground_truth = card_data.get("ground_truth_flaws", [])
    flaw_types = [f.get("flaw_type") for f in ground_truth]
    
    if "missing_field" in flaw_types:
        return grade_missing_fields(action, ground_truth)
    elif "license_conflict" in flaw_types:
        return grade_license_conflict(action, ground_truth)
    elif "benchmark_fraud" in flaw_types:
        return grade_benchmark_fraud(action, ground_truth)
    else:
        return AuditReward(value=0.0, reason="Unknown flaw type", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)

