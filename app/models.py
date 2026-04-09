from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────
# ACTION — what the agent sends
# ─────────────────────────────────────────────

class AuditAction(BaseModel):
    """
    A single audit finding submitted by the agent.
    The agent calls step() with one of these per turn.
    """
    pillar: str = Field(
        description="Which audit pillar: model_card | dataset_qc | rl_reward | tool_tester"
    )
    finding_type: str = Field(
        description=(
            "Category of the finding. Valid values: "
            "missing_field, wrong_value, license_conflict, "
            "duplicate, near_duplicate, train_test_leakage, "
            "sparse_reward, reward_hacking, broken_verifier, "
            "no_docstring, no_type_hints, no_return_annotation, "
            "silent_failure, dangerous_chain"
        )
    )
    target_field: str = Field(
        description="The specific field, column, function, or component where the issue exists"
    )
    description: str = Field(
        description="Agent's natural language explanation of the finding"
    )
    severity: int = Field(
        ge=0, le=3,
        description="0=info, 1=warning, 2=error, 3=critical"
    )


# ─────────────────────────────────────────────
# OBSERVATION — what the agent sees
# ─────────────────────────────────────────────

class AuditObservation(BaseModel):
    """
    Everything the agent can observe at the start of an episode
    or after each step. Returned by reset() and embedded in StepResult.
    """
    artifact_type: str = Field(
        description="Type of artifact: model_card | dataset | rl_config | tool"
    )
    content: str = Field(
        description="Full text content of the artifact to audit"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured metadata — varies by artifact type"
    )
    task_id: str = Field(
        description="The active task identifier from openenv.yaml"
    )
    instructions: str = Field(
        description="Natural language description of what the agent must find"
    )
    step_number: int = Field(
        default=0,
        description="Current step index within the episode"
    )
    max_steps: int = Field(
        description="Maximum number of steps allowed in this episode"
    )
    findings_so_far: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All findings the agent has submitted so far this episode"
    )
    flaws_found_count: int = Field(
        default=0,
        description="Number of ground-truth flaws correctly identified so far"
    )
    total_flaws: int = Field(
        description="Total number of ground-truth flaws to find in this episode"
    )


# ─────────────────────────────────────────────
# REWARD — per-step reward signal
# ─────────────────────────────────────────────

class AuditReward(BaseModel):
    """
    Reward signal returned after each step.
    Partial rewards are given per correct finding — not binary end-of-episode.
    """
    value: float = Field(
        gt=0.0, lt=1.0,
        description="Reward for this step. Range: 0.0 (no credit) to 1.0 (full credit)"
    )
    reason: str = Field(
        description="Human-readable explanation of why this reward was given"
    )
    finding_matched: Optional[str] = Field(
        default=None,
        description="ID of the ground-truth flaw that was matched, or null if none"
    )
    is_false_positive: bool = Field(
        default=False,
        description="True if the finding did not match any ground-truth flaw"
    )
    penalty_applied: float = Field(
        default=0.0,
        description="Any penalty deducted (false positive or step overflow)"
    )
    cumulative_score: float = Field(
        default=0.5,
        description="Running total score for this episode so far"
    )


# ─────────────────────────────────────────────
# STEP RESULT — what step() returns
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Full response from POST /step.
    Matches the OpenEnv spec: observation, reward, done, info.
    """
    observation: AuditObservation
    reward: float = Field(
        description="Scalar reward for this step, extracted from AuditReward.value"
    )
    done: bool = Field(
        description="True when the episode is complete"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information — audit_reward details, episode stats"
    )


# ─────────────────────────────────────────────
# RESET RESULT — what reset() returns
# ─────────────────────────────────────────────

class ResetResult(BaseModel):
    """
    Response from POST /reset.
    Returns the initial observation for the new episode.
    """
    observation: AuditObservation
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task metadata — pillar, difficulty, max_steps, total_flaws"
    )


# ─────────────────────────────────────────────
# RESET REQUEST — optional body for /reset
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    """
    Optional request body for POST /reset.
    If task_id is omitted, environment picks a random task.
    """
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID from openenv.yaml. If null, a random task is selected."
    )

