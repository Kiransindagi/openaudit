"""
OpenAudit Environment - Complete with All 4 Pillars
"""
import json
import uuid
from typing import Dict, Any, Optional, List
from app.models import AuditObservation, AuditAction, AuditReward
from app.pillars.model_card import grade_model_card, load_card
from app.pillars.dataset_qc import grade_dataset, load_dataset
from app.pillars.rl_reward import grade_reward, load_rl_config
from app.pillars.tool_tester import grade_tool, load_tool

class OpenAuditEnv:
    def __init__(self):
        self.current_episode_id = None
        self.current_pillar = None
        self.current_task_id = None
        self.current_artifact = None
        self.step_number = 0
        self.findings_so_far = []
        self.max_steps = 10
        self.completed = False
        self.total_reward = 0.0
        self.flaws_found_count = 0
        
        # Complete task registry - all 4 pillars
        self.tasks = {
            # Model Card Pillar
            "model_card_easy": {"pillar": "model_card", "artifact_id": "card_0", "max_steps": 8},
            "model_card_medium": {"pillar": "model_card", "artifact_id": "card_1", "max_steps": 10},
            "model_card_hard": {"pillar": "model_card", "artifact_id": "card_2", "max_steps": 12},
            
            # Dataset QC Pillar
            "dataset_qc_easy": {"pillar": "dataset_qc", "artifact_id": "dataset_0", "max_steps": 8},
            "dataset_qc_medium": {"pillar": "dataset_qc", "artifact_id": "dataset_1", "max_steps": 10},
            "dataset_qc_hard": {"pillar": "dataset_qc", "artifact_id": "dataset_2", "max_steps": 12},
            
            # RL Reward Pillar
            "rl_reward_easy": {"pillar": "rl_reward", "artifact_id": "rl_0", "max_steps": 8},
            "rl_reward_medium": {"pillar": "rl_reward", "artifact_id": "rl_1", "max_steps": 10},
            "rl_reward_hard": {"pillar": "rl_reward", "artifact_id": "rl_2", "max_steps": 12},
            
            # Tool Tester Pillar
            "tool_tester_easy": {"pillar": "tool_tester", "artifact_id": "tool_0", "max_steps": 8},
            "tool_tester_medium": {"pillar": "tool_tester", "artifact_id": "tool_1", "max_steps": 10},
            "tool_tester_hard": {"pillar": "tool_tester", "artifact_id": "tool_2", "max_steps": 12},
        }
    
    def reset(self, task_id: str = None) -> AuditObservation:
        print(f"[DEBUG] reset() called with task_id={task_id}")
        
        if not task_id or task_id not in self.tasks:
            task_id = "model_card_easy"
        
        task_config = self.tasks[task_id]
        
        self.current_episode_id = f"ep_{uuid.uuid4().hex[:6]}"
        self.current_pillar = task_config["pillar"]
        self.current_task_id = task_id
        self.step_number = 0
        self.findings_so_far = []
        self.completed = False
        self.total_reward = 0.0
        self.flaws_found_count = 0
        self.max_steps = task_config["max_steps"]
        
        artifact_id = task_config["artifact_id"]
        
        # Load artifact based on pillar
        if self.current_pillar == "model_card":
            self.current_artifact = load_card(artifact_id)
            content = self.current_artifact.get("card_text", "")
            metadata = self.current_artifact.get("metadata", {})
            total_flaws = len(self.current_artifact.get("ground_truth_flaws", []))
            instructions = "Find missing required fields: license, evaluation results, and CO2 emissions."
        elif self.current_pillar == "dataset_qc":
            self.current_artifact = load_dataset(artifact_id)
            content = "Dataset loaded for QC inspection"
            metadata = self.current_artifact.get("metadata", {})
            total_flaws = len([f for f in self.current_artifact.get("ground_truth_flaws", []) if f.get("type") == "null_values"])
            instructions = "Find null values in the dataset columns."
        elif self.current_pillar == "rl_reward":
            self.current_artifact = load_rl_config(artifact_id)
            content = "RL config loaded for reward auditing"
            metadata = self.current_artifact.get("metadata", {})
            total_flaws = len([f for f in self.current_artifact.get("ground_truth_flaws", []) if f.get("type") == "sparse_reward"])
            instructions = "Identify sparse reward issues."
        else:  # tool_tester
            self.current_artifact = load_tool(artifact_id)
            content = "Tool loaded for security analysis"
            metadata = self.current_artifact.get("metadata", {})
            total_flaws = len([f for f in self.current_artifact.get("ground_truth_flaws", []) if f.get("type") == "code_quality"])
            instructions = "Find code quality issues: missing docstrings, type hints, and return annotations."
        
        return AuditObservation(
            artifact_type=self.current_pillar,
            content=content,
            metadata=metadata,
            step_number=self.step_number,
            findings_so_far=self.findings_so_far,
            max_steps=self.max_steps,
            task_id=self.current_task_id,
            instructions=instructions,
            flaws_found_count=self.flaws_found_count,
            total_flaws=total_flaws
        )
    
    def step(self, action: AuditAction) -> tuple:
        if self.completed:
            return self._get_observation(), self.total_reward, True, {"error": "Episode completed"}
        
        if self.step_number >= self.max_steps:
            self.completed = True
            return self._get_observation(), self.total_reward, True, {"error": "Max steps reached"}
        
        if action.pillar != self.current_pillar:
            return self._get_observation(), -0.2, False, {"error": f"Wrong pillar. Expected {self.current_pillar}"}
        
        reward_obj = self._grade_action(action)
        reward_value = reward_obj.value
        
        if reward_obj.finding_matched and not reward_obj.is_false_positive:
            self.flaws_found_count += 1
        
        self.findings_so_far.append({
            "step": self.step_number,
            "action": action.dict(),
            "reward": reward_value,
            "reason": reward_obj.reason
        })
        
        self.total_reward += reward_value
        self.step_number += 1
        
        total_flaws = self._get_total_flaws()
        if self.flaws_found_count >= total_flaws and total_flaws > 0:
            self.completed = True
            self.total_reward += 0.2
        
        if self.step_number >= self.max_steps:
            self.completed = True
        
        return self._get_observation(), self.total_reward, self.completed, {}
    
    def _grade_action(self, action: AuditAction) -> AuditReward:
        if self.current_pillar == "model_card":
            return grade_model_card(action, self.current_artifact)
        elif self.current_pillar == "dataset_qc":
            return grade_dataset(action, self.current_artifact)
        elif self.current_pillar == "rl_reward":
            return grade_reward(action, self.current_artifact)
        elif self.current_pillar == "tool_tester":
            return grade_tool(action, self.current_artifact)
        else:
            return AuditReward(value=0.0, reason="Unknown pillar", finding_matched=None, is_false_positive=True, penalty_applied=0.0, cumulative_score=0.0)
    
    def _get_observation(self) -> AuditObservation:
        total_flaws = self._get_total_flaws()
        return AuditObservation(
            artifact_type=self.current_pillar,
            content="",
            metadata={},
            step_number=self.step_number,
            findings_so_far=self.findings_so_far,
            max_steps=self.max_steps,
            task_id=self.current_task_id,
            instructions="",
            flaws_found_count=self.flaws_found_count,
            total_flaws=total_flaws
        )
    
        def _get_total_flaws(self) -> int:
        if not self.current_artifact:
            return 0
        if self.current_pillar == "model_card":
            return len(self.current_artifact.get("ground_truth_flaws", []))
        elif self.current_pillar == "dataset_qc":
            # For dataset_qc, we count null_values, duplicates, test_leakage separately?
            # Simplified: return number of flaws
            return len(self.current_artifact.get("ground_truth_flaws", []))
        elif self.current_pillar == "rl_reward":
            # Count any flaw in ground_truth_flaws
            return len(self.current_artifact.get("ground_truth_flaws", []))
        elif self.current_pillar == "tool_tester":
            return len(self.current_artifact.get("ground_truth_flaws", []))
        return 0
    def get_state(self) -> dict:
        return {
            "episode_id": self.current_episode_id,
            "current_pillar": self.current_pillar,
            "current_task": self.current_task_id,
            "step_number": self.step_number,
            "findings_so_far": len(self.findings_so_far),
            "total_reward": self.total_reward,
            "max_steps": self.max_steps,
            "completed": self.completed
        }

_env_instance = None

def get_env():
    global _env_instance
    if _env_instance is None:
        _env_instance = OpenAuditEnv()
    return _env_instance



