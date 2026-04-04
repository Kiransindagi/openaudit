from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path

# Create the FastAPI app instance - THIS IS REQUIRED
app = FastAPI(title="OpenAudit", version="0.1.0")

# Import your models
from app.models import AuditAction, AuditObservation, AuditReward

# State storage (in-memory for now)
audit_state = {
    "episode_id": None,
    "current_pillar": 0,
    "current_task": 0,
    "observations": [],
    "actions": [],
    "rewards": [],
    "completed": False
}

DATA_DIR = Path("data")

@app.post("/reset")
def reset_episode():
    """Reset the audit state machine for a new episode."""
    global audit_state
    import random
    audit_state = {
        "episode_id": f"ep_{random.randint(1000, 9999)}",
        "current_pillar": 0,
        "current_task": 0,
        "observations": [],
        "actions": [],
        "rewards": [],
        "completed": False
    }
    return {"status": "reset", "episode_id": audit_state["episode_id"]}

@app.post("/step")
def step(action: AuditAction):
    """Execute one step in the audit state machine."""
    global audit_state
    
    if audit_state["completed"]:
        raise HTTPException(status_code=400, detail="Episode completed, reset first")
    
    # Store action
    audit_state["actions"].append(action.dict())
    
    # Get current pillar and task
    pillars = ["model_card", "dataset_qc", "rl_reward", "tool_tester"]
    pillar_idx = audit_state["current_pillar"]
    task_idx = audit_state["current_task"]
    
    if pillar_idx >= len(pillars):
        audit_state["completed"] = True
        return {"status": "complete", "total_reward": sum(audit_state["rewards"])}
    
    pillar = pillars[pillar_idx]
    
    # Generate observation
    obs = generate_observation(pillar, task_idx)
    audit_state["observations"].append(obs)
    
    # Calculate reward (placeholder)
    reward = calculate_reward(pillar, task_idx, action)
    audit_state["rewards"].append(reward.value)
    
    # Advance state
    audit_state["current_task"] += 1
    if audit_state["current_task"] >= 3:  # 3 tasks per pillar
        audit_state["current_task"] = 0
        audit_state["current_pillar"] += 1
    
    # Check completion
    if audit_state["current_pillar"] >= len(pillars):
        audit_state["completed"] = True
    
    return {
        "observation": obs,
        "reward": reward.dict(),
        "done": audit_state["completed"],
        "info": {
            "pillar": pillar,
            "pillar_idx": pillar_idx,
            "task_idx": task_idx
        }
    }

@app.get("/state")
def get_state():
    """Get current state machine status."""
    return audit_state

def generate_observation(pillar: str, task_id: int) -> Dict[str, Any]:
    """Load target data for current pillar/task."""
    data_map = {
        "model_card": ("model_cards", f"card_{task_id}.json"),
        "dataset_qc": ("datasets", f"dataset_{task_id}.json"),
        "rl_reward": ("rl_configs", f"rl_{task_id}.json"),
        "tool_tester": ("tools", f"tool_{task_id}.json")
    }
    
    folder, filename = data_map.get(pillar, ("", ""))
    filepath = DATA_DIR / folder / filename
    
    content = {}
    if filepath.exists():
        with open(filepath) as f:
            content = json.load(f)
    
    return {
        "artifact_type": pillar,
        "content": json.dumps(content),
        "metadata": content,
        "step_number": audit_state["current_task"],
        "findings_so_far": audit_state["observations"],
        "max_steps": 12,
        "task_id": f"{pillar}_{task_id}",
        "instructions": f"Audit {pillar} task {task_id}"
    }

def calculate_reward(pillar: str, task_id: int, action: AuditAction) -> AuditReward:
    """Calculate reward based on action quality."""
    # Placeholder grading logic - will integrate actual graders from pillars/
    score = 0.5
    
    if action.finding_type:
        score = 0.7
    
    return AuditReward(
        value=round(score, 3),
        reason=f"Processed {action.finding_type} for {pillar} task {task_id}",
        finding_matched=None,
        is_false_positive=False
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
