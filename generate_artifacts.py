"""
Dynamic Artifact Generator - Creates infinite variations of flaws
Prevents agent overfitting to static JSON files
"""
import json
import random
import os
from pathlib import Path

def generate_model_card(card_id, seed=None):
    """Generate a model card with random missing fields"""
    if seed:
        random.seed(seed)
    
    all_fields = ["license", "eval_results", "co2_emitted", "base_model", "training_data", "limitations"]
    missing_count = random.randint(2, 4)
    missing_fields = random.sample(all_fields, missing_count)
    
    # Build card text based on missing fields
    card_text = "# Model Card: Dynamic-Model-{card_id}\n\n## Model Details\n- Architecture: Transformer\n- Parameters: {params}B\n"
    
    license_text = "[MISSING]" if "license" in missing_fields else "MIT"
    eval_text = "[MISSING]" if "eval_results" in missing_fields else '{"MMLU": 72.5}'
    co2_text = "[MISSING]" if "co2_emitted" in missing_fields else "45.2 kg"
    
    card_text += f"\n## License\n{license_text}\n"
    card_text += f"\n## Evaluation Results\n{eval_text}\n"
    card_text += f"\n## Environmental Impact\nCO2 Emitted: {co2_text}\n"
    
    return {
        "card_id": f"card_{card_id}",
        "card_text": card_text,
        "metadata": {
            "model_name": f"Dynamic-Model-{card_id}",
            "parameters": f"{random.randint(1, 70)}B",
            "missing_fields": missing_fields
        },
        "ground_truth_flaws": [
            {
                "flaw_type": "missing_field",
                "fields": missing_fields,
                "severity": "error"
            }
        ],
        "difficulty": "easy" if missing_count <= 2 else "medium" if missing_count <= 3 else "hard"
    }

def generate_dataset(dataset_id, seed=None):
    """Generate a dataset with random null values"""
    if seed:
        random.seed(seed)
    
    rows = []
    null_columns = random.sample(["colA", "colB", "colC", "colD"], k=random.randint(2, 3))
    null_positions = random.sample(range(100), k=random.randint(5, 15))
    
    for i in range(100):
        row = {}
        for col in ["colA", "colB", "colC", "colD"]:
            if col in null_columns and i in null_positions:
                row[col] = None
            else:
                row[col] = f"value_{i}"
        row["id"] = i
        row["split"] = "train" if i < 80 else "test"
        rows.append(row)
    
    return {
        "dataset_id": f"dataset_{dataset_id}",
        "dataset": rows,
        "metadata": {
            "rows": 100,
            "columns": ["colA", "colB", "colC", "colD", "id", "split"],
            "null_columns": null_columns,
            "null_count": len(null_positions)
        },
        "ground_truth_flaws": [
            {
                "type": "null_values",
                "columns": null_columns,
                "total_nulls": len(null_positions)
            }
        ],
        "difficulty": "easy"
    }

# Generate 20 dynamic artifacts for each pillar
print("Generating dynamic artifacts...")
for i in range(20, 30):  # Generate additional 10
    card = generate_model_card(i, seed=i)
    with open(f"data/model_cards/dynamic_card_{i}.json", "w") as f:
        json.dump(card, f, indent=2)
    
    dataset = generate_dataset(i, seed=i)
    with open(f"data/datasets/dynamic_dataset_{i}.json", "w") as f:
        json.dump(dataset, f, indent=2)

print(f"Generated {len(os.listdir('data/model_cards'))} model cards")
print(f"Generated {len(os.listdir('data/datasets'))} datasets")
