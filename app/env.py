import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
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

        self.tasks = {
            "model_card_easy": {"pillar": "model_card", "artifact_id": "card_0", "max_steps": 8},
            "model_card_medium": {"pillar": "model_card", "artifact_id": "card_1", "max_steps": 10},
            "model_card_hard": {"pillar": "model_card", "artifact_id": "card_2", "max_steps": 12},
            "dataset_qc_easy": {"pillar": "dataset_qc", "artifact_id": "dataset_0", "max_steps": 8},
            "dataset_qc_medium": {"pillar": "dataset_qc", "artifact_id": "dataset_1", "max_steps": 10},
            "dataset_qc_hard": {"pillar": "dataset_qc", "artifact_id": "dataset_2", "max_steps": 12},
            "rl_reward_easy": {"pillar": "rl_reward", "artifact_id": "rl_0", "max_steps": 8},
            "rl_reward_medium": {"pillar": "rl_reward", "artifact_id": "rl_1", "max_steps": 10},
            "rl_reward_hard": {"pillar": "rl_reward", "artifact_id": "rl_2", "max_steps": 12},
            "tool_tester_easy": {"pillar": "tool_tester", "artifact_id": "tool_0", "max_steps": 8},
            "tool_tester_medium": {"pillar": "tool_tester", "artifact_id": "tool_1", "max_steps": 10},
            "tool_tester_hard": {"pillar": "tool_tester", "artifact_id": "tool_2", "max_steps": 12},
            "model_card_audit_chain": {"pillar": "model_card", "artifact_id": "card_0", "max_steps": 15}
        }

    def reset(self, task_id: str = None) -> AuditObservation:
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

        if self.current_pillar == "model_card":
            self.current_artifact = load_card(artifact_id)
            content = self.current_artifact.get("card_text", "")
            metadata = self.current_artifact.get("metadata", {})
            instructions = "Find missing required fields."
        elif self.current_pillar == "dataset_qc":
            self.current_artifact = load_dataset(artifact_id)
            dataset = self.current_artifact.get("dataset", [])
            content = json.dumps({"sample_rows": dataset[:20], "total_rows": len(dataset)}, indent=2)
            metadata = self.current_artifact.get("metadata", {})
            instructions = "Find null values in the dataset columns."
        elif self.current_pillar == "rl_reward":
            self.current_artifact = load_rl_config(artifact_id)
            content = json.dumps(self.current_artifact.get("trajectory_log", self.current_artifact), indent=2)
            metadata = {"config": self.current_artifact.get("config", {})}
            instructions = "Identify sparse reward issues."
        else:
            self.current_artifact = load_tool(artifact_id)
            content = self.current_artifact.get("tool_code", "")
            metadata = self.current_artifact.get("metadata", {})
            instructions = "Find code quality issues."

        total_flaws = self._get_total_flaws()
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
            return self._get_observation(), 0.5, True, {"error": "Done"}

        if self.step_number >= self.max_steps:
            self.completed = True
            normalized = self._get_normalized_score()
            return self._get_observation(), normalized, True, {"error": "Max steps"}

        if action.pillar != self.current_pillar:
            return self._get_observation(), 0.3, False, {"error": "Wrong pillar"}

        try:
            reward_obj = self._grade_action(action)
            reward_value = reward_obj.value
        except:
            reward_value = 0.5

        if reward_value <= 0.0:
            reward_value = 0.001
        elif reward_value >= 1.0:
            reward_value = 0.999

        self.findings_so_far.append({"step": self.step_number, "action": action.dict(), "reward": reward_value})
        self.total_reward += reward_value
        self.step_number += 1

        total_flaws = self._get_total_flaws()
        if self.flaws_found_count >= total_flaws and total_flaws > 0:
            self.completed = True

        if self.step_number >= self.max_steps:
            self.completed = True

        if self.completed:
            normalized = self._get_normalized_score()
            return self._get_observation(), normalized, self.completed, {}
        else:
            return self._get_observation(), reward_value, self.completed, {}

    def _get_normalized_score(self) -> float:
        max_possible = self.max_steps * 0.5
        if max_possible > 0:
            normalized = self.total_reward / max_possible
        else:
            normalized = 0.5
        if normalized <= 0.0:
            return 0.001
        if normalized >= 1.0:
            return 0.999
        return normalized

    def _grade_action(self, action: AuditAction) -> AuditReward:
        if self.current_pillar == "model_card":
            return grade_model_card(action, self.current_artifact)
        elif self.current_pillar == "dataset_qc":
            return grade_dataset(action, self.current_artifact)
        elif self.current_pillar == "rl_reward":
            return grade_reward(action, self.current_artifact)
        else:
            return grade_tool(action, self.current_artifact)

    def _get_observation(self) -> AuditObservation:
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
            total_flaws=self._get_total_flaws()
        )

    def _get_total_flaws(self) -> int:
        try:
            if not self.current_artifact:
                return 1
            flaws = self.current_artifact.get("ground_truth_flaws", [])
            if not flaws:
                return 1
            return len(flaws)
        except:
            return 1

    def get_state(self) -> dict:
        return {
            "episode_id": self.current_episode_id,
            "current_pillar": self.current_pillar,
            "current_task": self.current_task_id,
            "step_number": self.step_number,
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
