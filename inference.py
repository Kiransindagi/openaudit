"""
OpenAudit Baseline Agent - Uses real LLM (OpenAI API) for decision making
"""
import os
import json
import requests
from openai import OpenAI

# Environment variables
ENV_API_URL = os.environ.get("API_BASE_URL", "https://kiransin-openaudit.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # Real OpenAI key
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # For environment access if needed

# Initialize OpenAI client with real API
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.openai.com/v1"
)

TASKS = ["model_card_easy", "dataset_qc_easy", "rl_reward_easy"]
MAX_STEPS_PER_TASK = 5

def get_llm_action(observation, step, previous_actions):
    """Call real LLM to decide next action"""
    system_prompt = """You are an AI auditor. Your job is to find flaws in AI artifacts.
You will receive an artifact (model card, dataset, reward function, or tool) and instructions.
Output ONLY a JSON action with these fields:
- pillar: the artifact type (model_card, dataset_qc, rl_reward, tool_tester)
- finding_type: the type of flaw (missing_field, null_values, sparse_reward, code_quality, etc.)
- target_field: the specific field or component
- description: a brief explanation
- severity: 0-3

No other text, just the JSON."""

    user_prompt = f"""Artifact type: {observation['artifact_type']}
Instructions: {observation['instructions']}
Artifact content (first 2000 chars): {observation['content'][:2000]}
Current step: {step}
Previous findings submitted: {json.dumps(previous_actions, indent=2)}

What flaw should you report next? Output JSON."""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        action_text = response.choices[0].message.content
        # Extract JSON
        if "```json" in action_text:
            action_text = action_text.split("```json")[1].split("```")[0]
        elif "```" in action_text:
            action_text = action_text.split("```")[1].split("```")[0]
        return json.loads(action_text.strip())
    except Exception as e:
        print(f"LLM error: {e}, using fallback")
        # Fallback to simple rule-based
        pillar = observation['artifact_type']
        if pillar == "model_card":
            return {"pillar": pillar, "finding_type": "missing_field", "target_field": "license", "description": "Missing license field", "severity": 2}
        elif pillar == "dataset_qc":
            return {"pillar": pillar, "finding_type": "null_values", "target_field": "columns", "description": "Found null values", "severity": 2}
        else:
            return {"pillar": pillar, "finding_type": "sparse_reward", "target_field": "reward_function", "description": "Sparse reward detected", "severity": 2}

def run_task(task_id):
    print(f"[START] task={task_id} env=openaudit model={MODEL_NAME}", flush=True)

    # Reset environment
    resp = requests.post(f"{ENV_API_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        print(f"[STEP] step=0 action={{}} reward=0.0000 done=True error=Reset failed", flush=True)
        print(f"[END] success=False steps=1 score=0.0000 rewards=0.0000", flush=True)
        return 0.0

    observation = resp.json().get("observation", {})
    step = 0
    total_reward = 0.0
    step_rewards = []
    done = False
    previous_actions = []

    while step < MAX_STEPS_PER_TASK and not done:
        action = get_llm_action(observation, step, previous_actions)
        
        step_resp = requests.post(f"{ENV_API_URL}/step", json=action)
        if step_resp.status_code != 200:
            print(f"[STEP] step={step} action={json.dumps(action)} reward=0.0000 done=True error=Step failed", flush=True)
            break

        result = step_resp.json()
        reward = result.get("reward", 0.0)
        total_reward += reward
        step_rewards.append(reward)
        done = result.get("done", False)
        
        print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={done} error=", flush=True)
        
        previous_actions.append(action)
        step += 1
        if done:
            break

    task_score = min(total_reward / MAX_STEPS_PER_TASK, 1.0)
    rewards_str = ",".join([f"{r:.4f}" for r in step_rewards])
    print(f"[END] success=True steps={step} score={task_score:.4f} rewards={rewards_str}", flush=True)
    return task_score

def main():
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Using fallback actions only.")
    total_score = 0.0
    for task in TASKS:
        score = run_task(task)
        total_score += score
    print(f"Overall score: {total_score / len(TASKS):.4f}")

if __name__ == "__main__":
    main()
