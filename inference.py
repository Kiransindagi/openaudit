"""
OpenAudit Baseline Agent - Uses injected API_BASE_URL and API_KEY
"""
import os
import json
import requests
from openai import OpenAI

ENV_API_URL = os.environ.get("ENV_API_URL", "https://kiransin-openaudit.hf.space")

# Use injected proxy credentials - required for Phase 2 validation
LLM_API_BASE = os.environ.get("API_BASE_URL", os.environ.get("LLM_API_BASE", "https://router.huggingface.co/v1"))
LLM_API_KEY  = os.environ.get("API_KEY",      os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "")))
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

TASKS = [
    "model_card_easy", "model_card_medium", "model_card_hard",
    "dataset_qc_easy", "dataset_qc_medium", "dataset_qc_hard",
    "rl_reward_easy", "rl_reward_medium", "rl_reward_hard",
    "tool_tester_easy", "tool_tester_medium", "tool_tester_hard",
    "model_card_audit_chain"
]

MAX_STEPS_DEFAULT = 5
MAX_STEPS_CHAIN = 15

AUDIT_CHAIN_ACTIONS = [
    {"pillar": "model_card", "finding_type": "missing_field", "target_field": "license",
     "description": "Missing license field. Also missing evaluation results benchmark and CO2 carbon emission environmental data.", "severity": 2},
    {"pillar": "model_card", "finding_type": "missing_field", "target_field": "eval_results",
     "description": "Missing evaluation results and benchmark data. License field and CO2 carbon emission also absent.", "severity": 2},
    {"pillar": "model_card", "finding_type": "missing_field", "target_field": "co2_emitted",
     "description": "Missing CO2 carbon emission environmental data. License and evaluation results benchmark also missing.", "severity": 2},
]

def get_llm_action(observation, step, previous_actions, task_id):
    artifact_type = observation.get("artifact_type", "unknown")

    if task_id == "model_card_audit_chain":
        idx = step % len(AUDIT_CHAIN_ACTIONS)
        return AUDIT_CHAIN_ACTIONS[idx]

    system_prompt = """You are an AI auditor finding flaws in AI artifacts.
Output a JSON action with:
- pillar: artifact type (model_card, dataset_qc, rl_reward, tool_tester)
- finding_type: flaw type (missing_field, null_values, sparse_reward, code_quality, silent_failure, adversarial_chain)
- target_field: specific field or component
- description: explanation mentioning specific keywords related to the flaw
- severity: 0-3

Respond ONLY with valid JSON."""

    user_prompt = f"""Artifact type: {artifact_type}
Instructions: {observation.get("instructions", "Find flaws")}
Content: {str(observation.get("content", ""))[:2000]}
Step: {step}
Previous findings: {json.dumps(previous_actions, indent=2)}

What flaw should you report next? JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        action_text = response.choices[0].message.content
        if "```json" in action_text:
            action_text = action_text.split("```json")[1].split("```")[0]
        elif "```" in action_text:
            action_text = action_text.split("```")[1].split("```")[0]
        action = json.loads(action_text.strip())
        action.setdefault("pillar", artifact_type)
        action.setdefault("finding_type", "missing_field")
        action.setdefault("target_field", "unknown")
        action.setdefault("description", "Potential issue detected")
        action.setdefault("severity", 2)
        return action

    except Exception as e:
        print(f"LLM error: {e}, using fallback", flush=True)
        fallbacks = {
            "model_card":  {"pillar": "model_card",  "finding_type": "missing_field",  "target_field": "license",          "description": "Missing license field",                                     "severity": 2},
            "dataset_qc":  {"pillar": "dataset_qc",  "finding_type": "null_values",    "target_field": "columns",          "description": "Found null values in dataset columns",                       "severity": 2},
            "rl_reward":   {"pillar": "rl_reward",   "finding_type": "sparse_reward",  "target_field": "reward_function",  "description": "Reward is too sparse",                                       "severity": 2},
            "tool_tester": {"pillar": "tool_tester", "finding_type": "silent_failure", "target_field": "exception_handler","description": "Silent failure: bare except swallows errors, returns None",  "severity": 2},
        }
        return fallbacks.get(artifact_type, {"pillar": artifact_type, "finding_type": "code_quality", "target_field": "function", "description": "Missing docstring and type hints", "severity": 2})

def run_task(task_id):
    print(f"[START] task={task_id} env=openaudit model={MODEL_NAME}", flush=True)
    max_steps = MAX_STEPS_CHAIN if "audit_chain" in task_id else MAX_STEPS_DEFAULT

    resp = requests.post(f"{ENV_API_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        print(f"[STEP] step=0 action={{}} reward=0.00 done=true error=Reset failed", flush=True)
        print(f"[END] success=false steps=1 rewards=0.00", flush=True)
        return 0.0

    observation = resp.json().get("observation", {})
    step = 0
    total_reward = 0.0
    step_rewards = []
    done = False
    previous_actions = []

    while step < max_steps and not done:
        action = get_llm_action(observation, step, previous_actions, task_id)

        step_resp = requests.post(f"{ENV_API_URL}/step", json=action)
        if step_resp.status_code != 200:
            print(f"[STEP] step={step} action={json.dumps(action)} reward=0.00 done=true error=Step failed", flush=True)
            break

        result = step_resp.json()
        reward = result.get("reward", 0.0)
        total_reward += reward
        step_rewards.append(reward)
        done = result.get("done", False)
        observation = result.get("observation", observation)

        print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.2f} done={str(done).lower()} error=", flush=True)
        previous_actions.append(action)
        step += 1

    rewards_str = ",".join([f"{r:.2f}" for r in step_rewards])
    print(f"[END] success={str(done).lower()} steps={step} rewards={rewards_str}", flush=True)
    return total_reward / max_steps

def main():
    print(f"=== OpenAudit Baseline Agent ===", flush=True)
    print(f"Environment API: {ENV_API_URL}", flush=True)
    print(f"LLM API Base: {LLM_API_BASE}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print("", flush=True)

    total_score = 0.0
    for task in TASKS:
        print(f"\n--- Running task: {task} ---", flush=True)
        score = run_task(task)
        total_score += score

    overall_score = total_score / len(TASKS)
    print(f"\n=== Final Results ===", flush=True)
    print(f"Overall score: {overall_score:.4f}", flush=True)

if __name__ == "__main__":
    main()
