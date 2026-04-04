"""
OpenAudit Environment - State Machine
Manages episode state, task routing, and grader orchestration
"""
from typing import Dict, Any, Optional, List
from app.models import AuditObservation, AuditAction, AuditReward
from app.pillars.model_card import grade_model_card, load_card
from app.pillars.dataset_qc import grade_dataset, load_dataset
from app.pillars.rl_reward import grade_reward, load_rl_config
from app.pillars.tool_tester import grade_tool, load_tool
import uuid

class OpenAuditEnv:
    """Main environment class for OpenAudit"""
    
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
        
        # Task registry
        self.tasks = {
            # Pillar 1: Model Card (3 tasks)
            "model_card_easy": {
                "pillar": "model_card",
                "difficulty": "easy",
                "card_id": "card_0",
                "max_steps": 8
            },
            "model_card_medium": {
                "pillar": "model_card", 
                "difficulty": "medium",
                "card_id": "card_1",
                "max_steps": 10
            },
            "model_card_hard": {
                "pillar": "model_card",
                "difficulty": "hard", 
                "card_id": "card_2",
                "max_steps": 12
            },
        }
    
    def reset(self, task_id: str = None) -> AuditObservation:
        """Reset environment and start new episode"""
        
        # Default to model_card_easy if no task specified
        if not task_id or task_id not in self.tasks:
            task_id = "model_card_easy"
        
        task_config = self.tasks[task_id]
        
        # Reset state
        self.current_episode_id = f"ep_{uuid.uuid4().hex[:6]}"
        self.current_pillar = task_config["pillar"]
        self.current_task_id = task_id
        self.step_number = 0
        self.findings_so_far = []
        self.completed = False
        self.total_reward = 0.0
        self.flaws_found_count = 0
        self.max_steps = task_config["max_steps"]
        
        # Load artifact based on pillar
        if self.current_pillar == "model_card":
            card_id = task_config["card_id"]
            self.current_artifact = load_card(card_id)
            content = self.current_artifact.get("card_text", "")
            metadata = self.current_artifact.get("metadata", {})
            instructions = self._get_instructions(task_id)
            total_flaws = len(self.current_artifact.get("ground_truth_flaws", []))
        else:
            # Placeholder for other pillars
            content = "Other pillars coming soon..."
            metadata = {}
            instructions = "Task not yet implemented"
            total_flaws = 0
        
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
    
    def step(self, action: AuditAction) -> tuple[AuditObservation, float, bool, dict]:
        """Process agent action and return next state"""
        
        if self.completed:
            return self._get_observation(), 0.0, True, {"error": "Episode already completed"}
        
        if self.step_number >= self.max_steps:
            self.completed = True
            return self._get_observation(), 0.0, True, {"error": "Max steps reached"}
        
        # Validate action pillar matches current task
        if action.pillar != self.current_pillar:
            return self._get_observation(), -0.2, False, {"error": f"Wrong pillar. Expected {self.current_pillar}"}
        
        # Grade the action
        reward_obj = self._grade_action(action)
        reward_value = reward_obj.value
        
        # Update flaws found count if finding matched
        if reward_obj.finding_matched and not reward_obj.is_false_positive:
            self.flaws_found_count += 1
        
        # Store finding
        finding = {
            "step": self.step_number,
            "action": action.dict(),
            "reward": reward_value,
            "reason": reward_obj.reason,
            "finding_matched": reward_obj.finding_matched,
            "is_false_positive": reward_obj.is_false_positive
        }
        self.findings_so_far.append(finding)
        self.total_reward += reward_value
        
        # Step penalty for inefficient agents
        step_penalty = 0.0
        if self.step_number > self.max_steps - 2:
            step_penalty = -0.05
            self.total_reward += step_penalty
        
        # Check if episode should end
        self.step_number += 1
        
        # End if max steps reached
        if self.step_number >= self.max_steps:
            self.completed = True
        
        # End if all flaws found
        if self.current_pillar == "model_card":
            expected_flaws = len(self.current_artifact.get("ground_truth_flaws", []))
            if self.flaws_found_count >= expected_flaws and expected_flaws > 0:
                self.completed = True
                # Bonus for completing early
                self.total_reward += 0.2
        
        observation = self._get_observation()
        
        return observation, self.total_reward, self.completed, {
            "step_penalty": step_penalty,
            "flaws_found": self.flaws_found_count,
            "total_flaws": len(self.current_artifact.get("ground_truth_flaws", [])) if self.current_artifact else 0
        }
    
    def _grade_action(self, action: AuditAction) -> AuditReward:
        """Route action to appropriate grader"""
        
        if self.current_pillar == "model_card":
            return grade_model_card(action, self.current_artifact)
        elif self.current_pillar == "dataset_qc":
            return grade_dataset(action, self.current_artifact)
        elif self.current_pillar == "rl_reward":
            return grade_reward(action, self.current_artifact)
        elif self.current_pillar == "tool_tester":
            return grade_tool(action, self.current_artifact)
        else:
            return AuditReward(
                value=0.0,
                reason=f"Pillar {self.current_pillar} not recognized",
                finding_matched=None,
                is_false_positive=True,
                penalty_applied=0.0,
                cumulative_score=self.total_reward
            )
    def _get_observation(self) -> AuditObservation:
        """Build current observation"""
        total_flaws = len(self.current_artifact.get("ground_truth_flaws", [])) if self.current_artifact else 0
        return AuditObservation(
            artifact_type=self.current_pillar,
            content=self.current_artifact.get("card_text", "") if self.current_artifact else "",
            metadata=self.current_artifact.get("metadata", {}) if self.current_artifact else {},
            step_number=self.step_number,
            findings_so_far=self.findings_so_far,
            max_steps=self.max_steps,
            task_id=self.current_task_id,
            instructions=self._get_instructions(self.current_task_id),
            flaws_found_count=self.flaws_found_count,
            total_flaws=total_flaws
        )
    
    def _get_instructions(self, task_id: str) -> str:
        """Get task-specific instructions for the agent"""
        instructions = {
            "model_card_easy": "Find missing required fields in this model card. Report which fields are missing: license, evaluation results, and/or CO2 emissions.",
            "model_card_medium": "Check for license conflicts. The parent model may have a different license than claimed. Identify the parent model and explain the license incompatibility.",
            "model_card_hard": "Verify benchmark claims against actual values. Find any discrepancies between claimed and actual benchmark scores.",
        }
        return instructions.get(task_id, "Find and report quality issues in this artifact.")
    
    def get_state(self) -> dict:
        """Return current episode state for GET /state endpoint"""
        total_flaws = len(self.current_artifact.get("ground_truth_flaws", [])) if self.current_artifact else 0
        return {
            "episode_id": self.current_episode_id,
            "current_pillar": self.current_pillar,
            "current_task": self.current_task_id,
            "step_number": self.step_number,
            "findings_so_far": len(self.findings_so_far),
            "flaws_found_count": self.flaws_found_count,
            "total_flaws": total_flaws,
            "total_reward": self.total_reward,
            "max_steps": self.max_steps,
            "completed": self.completed
        }

# Singleton instance
_env_instance = None

def get_env():
    """Get or create global environment instance"""
    global _env_instance
    if _env_instance is None:
        _env_instance = OpenAuditEnv()
    return _env_instance


