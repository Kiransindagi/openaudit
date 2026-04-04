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
    with open(filepath) as f:
        return json.load(f)

def grade_missing_fields(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Easy grader: Field completeness
    Score = |agent_found ∩ ground_truth| / |ground_truth|
    """
    # Extract missing fields from ground truth
    missing_fields: Set[str] = set()
    for flaw in ground_truth:
        if flaw["flaw_type"] == "missing_field":
            missing_fields.update(flaw.get("fields", []))
    
    if not missing_fields:
        return AuditReward(
            value=1.0,
            reason="No missing fields required",
            finding_matched=None,
            is_false_positive=False
        )
    
    # Parse agent's description for field mentions
    description = action.description.lower()
    agent_fields: Set[str] = set()
    
    # Map keywords to field names
    field_keywords = {
        "license": ["license", "licence"],
        "eval_results": ["eval", "evaluation", "eval_results", "benchmark", "mmlu", "truthfulqa"],
        "co2_emitted": ["co2", "carbon", "emission", "environmental", "climate"],
        "base_model": ["base model", "parent model", "derived from"],
        "training_data": ["training data", "dataset", "corpus"]
    }
    
    for field, keywords in field_keywords.items():
        if any(kw in description for kw in keywords):
            agent_fields.add(field)
    
    # Calculate intersection
    correct_matches = len(agent_fields & missing_fields)
    total_missing = len(missing_fields)
    
    score = correct_matches / total_missing if total_missing > 0 else 1.0
    
    # Penalty for false positives
    false_positives = len(agent_fields - missing_fields)
    score = max(0.0, score - (false_positives * 0.1))
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct_matches}/{total_missing} missing fields: {list(agent_fields & missing_fields)}",
        finding_matched=f"missing_field:{list(agent_fields & missing_fields)}" if correct_matches > 0 else None,
        is_false_positive=false_positives > 0
    )

def grade_license_conflict(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Medium grader: License conflict detection
    Check: parent model name + "license" keyword + conflict indicator
    """
    description = action.description.lower()
    
    # Find license conflict in ground truth
    conflict = None
    for flaw in ground_truth:
        if flaw["flaw_type"] == "license_conflict":
            conflict = flaw
            break
    
    if not conflict:
        return AuditReward(
            value=0.0,
            reason="No license conflict in this card",
            finding_matched=None,
            is_false_positive=True
        )
    
    parent_model = conflict.get("parent_model", "").lower()
    parent_license = conflict.get("parent_license", "").lower()
    
    # Check required components
    checks = {
        "has_license_keyword": "license" in description,
        "has_conflict": any(kw in description for kw in ["conflict", "incompatible", "violation", "mismatch", "gpl", "copyleft"]),
        "has_parent_name": parent_model.split("/")[-1].replace("-", " ") in description or parent_model in description
    }
    
    # Score: 0.3 per component, max 0.9
    score = sum([0.3 for v in checks.values() if v])
    
    return AuditReward(
        value=round(min(score, 1.0), 3),
        reason=f"License conflict detection: license={checks['has_license_keyword']}, conflict={checks['has_conflict']}, parent={checks['has_parent_name']}",
        finding_matched="license_conflict" if score >= 0.6 else None,
        is_false_positive=score < 0.3
    )

def grade_benchmark_fraud(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Hard grader: Benchmark fraud detection
    Check: correct benchmark mentioned + claimed number + actual number (within tolerance)
    """
    description = action.description.lower()
    
    # Find benchmark fraud in ground truth
    fraud = None
    for flaw in ground_truth:
        if flaw["flaw_type"] == "benchmark_fraud":
            fraud = flaw
            break
    
    if not fraud:
        return AuditReward(
            value=0.0,
            reason="No benchmark fraud in this card",
            finding_matched=None,
            is_false_positive=True
        )
    
    benchmark = fraud.get("benchmark", "").lower()
    claimed = fraud.get("claimed", 0.0)
    actual = fraud.get("actual", 0.0)
    tolerance = fraud.get("tolerance", 0.5)
    
    # Extract numbers from description
    numbers = [float(n) for n in re.findall(r'\d+\.?\d*', description)]
    
    # Check components
    has_benchmark = benchmark in description
    claimed_correct = any(abs(n - claimed) <= tolerance for n in numbers)
    actual_correct = any(abs(n - actual) <= tolerance for n in numbers)
    
    # Score: 0.3 benchmark + 0.35 claimed + 0.35 actual
    score = 0.0
    if has_benchmark:
        score += 0.3
    if claimed_correct:
        score += 0.35
    if actual_correct:
        score += 0.35
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Benchmark fraud: {benchmark} mentioned={has_benchmark}, claimed={claimed_correct} ({claimed}), actual={actual_correct} ({actual})",
        finding_matched="benchmark_fraud" if score >= 0.7 else None,
        is_false_positive=score < 0.3
    )

def grade_model_card(action: AuditAction, card_data: Dict[str, Any]) -> AuditReward:
    """
    Route to appropriate grader based on difficulty/ground truth
    """
    ground_truth = card_data.get("ground_truth_flaws", [])
    
    # Determine which grader to use based on flaw types present
    flaw_types = [f["flaw_type"] for f in ground_truth]
    
    if "missing_field" in flaw_types:
        return grade_missing_fields(action, ground_truth)
    elif "license_conflict" in flaw_types:
        return grade_license_conflict(action, ground_truth)
    elif "benchmark_fraud" in flaw_types:
        return grade_benchmark_fraud(action, ground_truth)
    else:
        return AuditReward(
            value=0.0,
            reason="Unknown flaw type",
            finding_matched=None,
            is_false_positive=True
        )

# ==================== UNIT TESTS ====================

def test_graders():
    """Test all graders: perfect action → 1.0, empty action → ~0.0"""
    print("Testing Model Card Graders...")
    
    test_cases = [
        ("card_0", "easy", "missing_field"),
        ("card_1", "medium", "license_conflict"),
        ("card_2", "hard", "benchmark_fraud")
    ]
    
    for card_id, difficulty, flaw_type in test_cases:
        print(f"\n=== Testing {card_id} ({difficulty}) ===")
        
        card = load_card(card_id)
        print(f"Flaws: {[f['flaw_type'] for f in card['ground_truth_flaws']]}")
        
        # Test 1: Empty action (should score ~0.0)
        empty_action = AuditAction(
            pillar="model_card",
            finding_type="",
            target_field="",
            description="",
            severity=0
        )
        empty_reward = grade_model_card(empty_action, card)
        print(f"Empty action: {empty_reward.value} (expected ~0.0)")
        assert empty_reward.value < 0.2, f"Empty should score ~0.0, got {empty_reward.value}"
        
        # Test 2: Perfect action (should score ~1.0)
        if difficulty == "easy":
            perfect_action = AuditAction(
                pillar="model_card",
                finding_type="missing_field",
                target_field="license",
                description="Missing license, eval_results, and co2_emitted fields",
                severity=2
            )
        elif difficulty == "medium":
            perfect_action = AuditAction(
                pillar="model_card",
                finding_type="license_conflict",
                target_field="license",
                description=f"License conflict with parent model {card['ground_truth_flaws'][0]['parent_model']} which uses GPL license",
                severity=3
            )
        else:  # hard
            flaw = card['ground_truth_flaws'][0]
            perfect_action = AuditAction(
                pillar="model_card",
                finding_type="benchmark_fraud",
                target_field=flaw['benchmark'],
                description=f"{flaw['benchmark']} benchmark fraud: claimed {flaw['claimed']} but actual is {flaw['actual']}",
                severity=3
            )
        
        perfect_reward = grade_model_card(perfect_action, card)
        print(f"Perfect action: {perfect_reward.value} (expected ~1.0)")
        assert perfect_reward.value >= 0.7, f"Perfect should score high, got {perfect_reward.value}"
        
        print(f"✅ {card_id} tests passed")
    
    print("\n✅ All grader tests passed!")

if __name__ == "__main__":
    test_graders()
