"""
OpenAudit Baseline Agent - Uses LLM for decision making
"""
import os
import json
import requests
from openai import OpenAI

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://kiransin-openaudit.hf.space")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize OpenAI client (REQUIRED by spec)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy"
)

# Tasks to run (3 tasks as required)
TASKS = ["model_card_easy", "dataset_qc_easy", "rl_reward_easy"]
MAX_STEPS_PER_TASK = 5
MAX_TOTAL_REWARD = len(TASKS) * MAX_STEPS_PER_TASK

def get_llm_action(observation, step, previous_actions):
    """Call LLM to decide next action based on observation"""
    
    system_prompt = """You are an AI auditor. Your job is to find flaws in AI artifacts.
You will receive an artifact (model card, dataset, reward function, or tool) and instructions.
You must output a JSON action with:
- pillar: the artifact type (model_card, dataset_qc, rl_reward, tool_tester)
- finding_type: the type of flaw (missing_field, null_values, sparse_reward, code_quality, etc.)
- target_field: the specific field or component
- description: a brief explanation
- severity: 0-3 (0=info, 1=warning, 2=error, 3=critical)

Respond ONLY with valid JSON, no other text."""

    user_prompt = f"""Artifact type: {observation.get('artifact_type', 'unknown')}
Instructions: {observation.get('instructions', 'Find flaws')}
Artifact content (first 2000 chars): {observation.get('content', '')[:2000]}
Current step: {step}
Previous findings submitted: {json.dumps(previous_actions, indent=2)}

Based on the instructions, what flaw should you report next? Output JSON."""

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
        
        # Parse JSON from response
        action_text = response.choices[0].message.content
        # Extract JSON if wrapped in markdown
        if "```json" in action_text:
            action_text = action_text.split("```json")[1].split("```")[0]
        elif "```" in action_text:
            action_text = action_text.split("```")[1].split("```")[0]
        action = json.loads(action_text.strip())
        
        # Ensure required fields are present
        if "pillar" not in action:
            action["pillar"] = observation.get('artifact_type', 'model_card')
        if "finding_type" not in action:
            action["finding_type"] = "missing_field"
        if "target_field" not in action:
            action["target_field"] = "unknown"
        if "description" not in action:
            action["description"] = "Potential issue detected"
        if "severity" not in action:
            action["severity"] = 2
            
        return action
        
    except Exception as e:
        print(f"LLM error: {e}, using fallback action")
        # Fallback: simple action based on pillar
        pillar = observation.get('artifact_type', 'model_card')
        if pillar == "model_card":
            return {"pillar": pillar, "finding_type": "missing_field", "target_field": "license", "description": "Missing license field", "severity": 2}
        elif pillar == "dataset_qc":
            return {"pillar": pillar, "finding_type": "null_values", "target_field": "columns", "description": "Found null values in dataset", "severity": 2}
        elif pillar == "rl_reward":
            return {"pillar": pillar, "finding_type": "sparse_reward", "target_field": "reward_function", "description": "Reward is too sparse", "severity": 2}
        else:
            return {"pillar": pillar, "finding_type": "code_quality", "target_field": "function", "description": "Missing docstring", "severity": 2}

def run_task(task_id):
    """Run a single audit task using LLM"""
    print(f"[START] task={task_id} env=openaudit model={MODEL_NAME}", flush=True)

    # Reset environment
    resp = requests.post(f"{API_BASE_URL}/reset", params={"task_id": task_id})
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
        # Get action from LLM (REAL LLM CALL)
        action = get_llm_action(observation, step, previous_actions)
        
        # Submit step to environment
        step_resp = requests.post(f"{API_BASE_URL}/step", json=action)
        if step_resp.status_code != 200:
            print(f"[STEP] step={step} action={json.dumps(action)} reward=0.0000 done=True error=Step failed", flush=True)
            break

        result = step_resp.json()
        reward = result.get("reward", 0.0)
        total_reward += reward
        step_rewards.append(reward)
        done = result.get("done", False)
        
        print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={str(done).lower()} error=", flush=True)
        
        previous_actions.append(action)
        step += 1
        
        if done:
            break

    # Calculate score for this task
    task_score = min(total_reward / MAX_STEPS_PER_TASK, 1.0)
    rewards_str = ",".join([f"{r:.4f}" for r in step_rewards])
    print(f"[END] success=True steps={step} score={task_score:.4f} rewards={rewards_str}", flush=True)
    return task_score

def main():
    """Run all tasks and compute final score"""
    print(f"=== OpenAudit Baseline Agent ===", flush=True)
    print(f"API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"Tasks: {TASKS}", flush=True)
    print("", flush=True)
    
    total_score = 0.0
    for task in TASKS:
        print(f"\n--- Running task: {task} ---", flush=True)
        score = run_task(task)
        total_score += score
    
    overall_score = total_score / len(TASKS)
    print(f"\n=== Final Results ===", flush=True)
    for i, task in enumerate(TASKS):
        print(f"{task}: score={score:.4f}", flush=True)
    print(f"Overall score: {overall_score:.4f}", flush=True)
    print(f"MAX_TOTAL_REWARD: {MAX_TOTAL_REWARD}", flush=True)

if __name__ == "__main__":
    main()