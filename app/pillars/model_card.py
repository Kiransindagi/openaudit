"""
Pillar 1: Model Card Auditing
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/model_cards")

def load_card(card_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{card_id}.json"
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def grade_missing_fields(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    missing_fields: Set[str] = set()
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "missing_field":
            missing_fields.update(flaw.get("fields", []))

    field_keywords = {
        "license":       ["license", "licence"],
        "eval_results":  ["eval", "evaluation", "benchmark"],
        "co2_emitted":   ["co2", "carbon", "emission"],
        "base_model":    ["base model", "parent model"],
        "training_data": ["training data", "dataset"]
    }

    agent_fields: Set[str] = set()
    for field, keywords in field_keywords.items():
        if any(kw in description for kw in keywords):
            agent_fields.add(field)

    correct = len(agent_fields & missing_fields)
    total = len(missing_fields)

    # Give full credit if ANY missing field keyword found
    score = 0.99 if correct > 0 else 0.01

    return AuditReward(
        value=score,
        reason=f"Found {correct}/{total} missing fields",
        finding_matched="missing_field" if correct > 0 else None,
        is_false_positive=correct == 0,
        penalty_applied=0.01,
        cumulative_score=score
    )

def grade_license_conflict(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    conflict = next((f for f in ground_truth if f.get("flaw_type") == "license_conflict"), None)
    if not conflict:
        return AuditReward(value=0.01, reason="No license conflict", finding_matched=None, is_false_positive=True, penalty_applied=0.01, cumulative_score=0.01)
    checks = {
        "license": "license" in description,
        "conflict": any(kw in description for kw in ["conflict", "incompatible", "violation", "gpl"]),
        "parent": conflict.get("parent_model", "").lower().split("/")[-1].replace("-", " ") in description
    }
    score = round(min(0.99, max(0.01, sum(0.33 for v in checks.values() if v))), 3)
    return AuditReward(value=score, reason="License conflict detection", finding_matched="license_conflict" if score >= 0.6 else None, is_false_positive=score < 0.3, penalty_applied=0.01, cumulative_score=score)

def grade_benchmark_fraud(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    description = action.description.lower()
    fraud = next((f for f in ground_truth if f.get("flaw_type") == "benchmark_fraud"), None)
    if not fraud:
        return AuditReward(value=0.01, reason="No benchmark fraud", finding_matched=None, is_false_positive=True, penalty_applied=0.01, cumulative_score=0.01)
    benchmark = fraud.get("benchmark", "").lower()
    claimed = fraud.get("claimed", 0.0)
    actual = fraud.get("actual", 0.0)
    numbers = [float(n) for n in re.findall(r"\d+\.?\d*", description)]
    score = 0.0
    if benchmark in description: score += 0.3
    if any(abs(n - claimed) <= 0.5 for n in numbers): score += 0.35
    if any(abs(n - actual) <= 0.5 for n in numbers): score += 0.35
    score = round(min(0.99, max(0.01, score)), 3)
    return AuditReward(value=score, reason="Benchmark fraud detection", finding_matched="benchmark_fraud" if score >= 0.6 else None, is_false_positive=score < 0.3, penalty_applied=0.01, cumulative_score=score)

def grade_model_card(action: AuditAction, card_data: Dict[str, Any]) -> AuditReward:
    ground_truth = card_data.get("ground_truth_flaws", [])
    flaw_types = [f.get("flaw_type") for f in ground_truth]
    if "missing_field" in flaw_types:
        return grade_missing_fields(action, ground_truth)
    elif "license_conflict" in flaw_types:
        return grade_license_conflict(action, ground_truth)
    elif "benchmark_fraud" in flaw_types:
        return grade_benchmark_fraud(action, ground_truth)
    return AuditReward(value=0.51, reason="Unknown flaw", finding_matched=None, is_false_positive=False, penalty_applied=0.01, cumulative_score=0.5)

