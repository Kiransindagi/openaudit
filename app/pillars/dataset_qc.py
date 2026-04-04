"""
Pillar 2: Dataset Quality Control - Corrected for 'type' field
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from app.models import AuditAction, AuditReward

DATA_DIR = Path("data/datasets")

def load_dataset(dataset_id: str) -> Dict[str, Any]:
    filepath = DATA_DIR / f"{dataset_id}.json"
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)

def grade_dataset(action: AuditAction, dataset_data: Dict[str, Any]) -> AuditReward:
    """Grader that looks for 'type' field in ground truth"""
    description = action.description.lower()
    ground_truth = dataset_data.get("ground_truth_flaws", [])
    
    # Find the flaw with type='null_values'
    target_flaw = None
    for flaw in ground_truth:
        if flaw.get("type") == "null_values":
            target_flaw = flaw
            break
    
    if target_flaw is None:
        return AuditReward(
            value=0.0,
            reason=f"No null_values flaw. Found types: {[f.get('type') for f in ground_truth]}",
            finding_matched=None,
            is_false_positive=True,
            penalty_applied=0.0,
            cumulative_score=0.0
        )
    
    # Check for null-related keywords
    if any(kw in description for kw in ["null", "missing", "empty"]):
        return AuditReward(
            value=0.8,
            reason="Correctly identified null values in dataset",
            finding_matched="null_values",
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.8
        )
    else:
        return AuditReward(
            value=0.2,
            reason="Partial credit - null values exist but not well described",
            finding_matched=None,
            is_false_positive=False,
            penalty_applied=0.0,
            cumulative_score=0.2
        )
