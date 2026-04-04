"""
Pillar 2: Dataset Quality Control Auditing
Deterministic graders for synthetic dataset quality issues
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/datasets")

def load_dataset(dataset_id: str) -> Dict[str, Any]:
    """Load dataset by ID (dataset_0 through dataset_9)"""
    filepath = DATA_DIR / f"{dataset_id}.json"
    with open(filepath) as f:
        return json.load(f)

def grade_null_detection(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Easy grader: Detect null values in dataset
    Score = correct_columns_detected / total_columns_with_nulls
    """
    description = action.description.lower()
    
    # Find null detection ground truth
    null_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "null_detection":
            null_flaw = flaw
            break
    
    if not null_flaw:
        return AuditReward(
            value=0.0,
            reason="No null detection required",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_columns = set(null_flaw.get("columns_with_nulls", []))
    total_columns = len(expected_columns)
    
    # Parse agent's description for column mentions
    agent_columns = set()
    for col in expected_columns:
        if col.lower() in description:
            agent_columns.add(col)
    
    correct_matches = len(agent_columns & expected_columns)
    score = correct_matches / total_columns if total_columns > 0 else 1.0
    
    # Penalty for false positives
    false_positives = len(agent_columns - expected_columns)
    score = max(0.0, score - (false_positives * 0.1))
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct_matches}/{total_columns} columns with nulls",
        finding_matched=f"null_detection:{list(agent_columns)}" if correct_matches > 0 else None,
        is_false_positive=false_positives > 0,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_duplicate_detection(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Medium grader: Detect duplicate rows
    Score = correct_duplicates_found / total_duplicates
    """
    description = action.description.lower()
    
    # Find duplicate detection ground truth
    dup_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "duplicate_detection":
            dup_flaw = flaw
            break
    
    if not dup_flaw:
        return AuditReward(
            value=0.0,
            reason="No duplicate detection required",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_duplicates = set(str(d) for d in dup_flaw.get("duplicate_pairs", []))
    total_duplicates = len(expected_duplicates)
    
    # Extract numbers from description (row indices)
    numbers = re.findall(r'\d+', description)
    agent_duplicates = set(numbers)
    
    correct_matches = len(agent_duplicates & expected_duplicates)
    score = correct_matches / total_duplicates if total_duplicates > 0 else 1.0
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Found {correct_matches}/{total_duplicates} duplicate pairs",
        finding_matched="duplicate_detection" if correct_matches > 0 else None,
        is_false_positive=correct_matches == 0,
        penalty_applied=0.0,
        cumulative_score=score
    )

def grade_leakage_detection(action: AuditAction, ground_truth: List[Dict]) -> AuditReward:
    """
    Hard grader: Detect train/test leakage
    Score = precision * recall on leaked test IDs
    """
    description = action.description.lower()
    
    # Find leakage ground truth
    leak_flaw = None
    for flaw in ground_truth:
        if flaw.get("flaw_type") == "train_test_leakage":
            leak_flaw = flaw
            break
    
    if not leak_flaw:
        return AuditReward(
            value=0.0,
            reason="No leakage detection required",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    expected_leaks = set(str(leak_flaw.get("leaked_test_ids", [])))
    total_leaks = len(expected_leaks)
    
    # Extract numbers from description
    numbers = set(re.findall(r'\d+', description))
    agent_leaks = numbers
    
    correct_matches = len(agent_leaks & expected_leaks)
    
    # Calculate precision and recall
    precision = correct_matches / len(agent_leaks) if len(agent_leaks) > 0 else 0
    recall = correct_matches / total_leaks if total_leaks > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return AuditReward(
        value=round(f1_score, 3),
        reason=f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1_score:.2f}",
        finding_matched="train_test_leakage" if correct_matches > 0 else None,
        is_false_positive=precision == 0,
        penalty_applied=0.0,
        cumulative_score=f1_score
    )

def grade_dataset(action: AuditAction, dataset_data: Dict[str, Any]) -> AuditReward:
    """Route to appropriate grader based on flaw type"""
    ground_truth = dataset_data.get("ground_truth_flaws", [])
    flaw_types = [f["flaw_type"] for f in ground_truth]
    
    if "null_detection" in flaw_types:
        return grade_null_detection(action, ground_truth)
    elif "duplicate_detection" in flaw_types:
        return grade_duplicate_detection(action, ground_truth)
    elif "train_test_leakage" in flaw_types:
        return grade_leakage_detection(action, ground_truth)
    else:
        return AuditReward(
            value=0.0,
            reason="Unknown flaw type",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )

def test_graders():
    """Test dataset QC graders"""
    print("Testing Dataset QC Graders...")
    print("Note: Create synthetic dataset files first!")
    print("✅ Placeholder - implement with actual data")

if __name__ == "__main__":
    test_graders()
