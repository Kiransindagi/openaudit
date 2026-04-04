import requests
import json

print('=== 1. Testing /reset ===')
r = requests.post('http://localhost:7860/reset?task_id=model_card_easy')
print(f'Status: {r.status_code}')
if r.status_code == 200:
    data = r.json()
    print('✓ Reset successful')
    # ResetResult has observation and info fields
    if 'observation' in data:
        obs = data['observation']
        print(f'  - Task: {obs.get("task_id")}')
        print(f'  - Total flaws: {obs.get("total_flaws")}')
        print(f'  - Max steps: {obs.get("max_steps")}')
    else:
        print(f'  - Response: {data}')
else:
    print(f'✗ Failed: {r.text}')
    exit()

print('\n=== 2. Testing /state ===')
r = requests.get('http://localhost:7860/state')
print(f'Status: {r.status_code}')
if r.status_code == 200:
    state = r.json()
    print(f'✓ State: episode={state.get("episode_id")}, completed={state.get("completed")}')

print('\n=== 3. Testing /step with PERFECT action ===')
action = {
    'pillar': 'model_card',
    'finding_type': 'missing_field',
    'target_field': 'license',
    'description': 'Missing license, eval_results, and co2_emitted fields',
    'severity': 2
}
r = requests.post('http://localhost:7860/step', json=action)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    result = r.json()
    print('✓ Step successful')
    print(f'  - Reward: {result.get("reward")}')
    print(f'  - Done: {result.get("done")}')
else:
    print(f'✗ Failed: {r.text}')

print('\n=== 4. Testing /step with EMPTY action ===')
action_empty = {
    'pillar': 'model_card',
    'finding_type': '',
    'target_field': '',
    'description': '',
    'severity': 0
}
r = requests.post('http://localhost:7860/step', json=action_empty)
print(f'Status: {r.status_code}')
if r.status_code == 200:
    result = r.json()
    print('✓ Step successful')
    print(f'  - Reward: {result.get("reward")}')
    print(f'  - Should be low (0.0-0.2): {result.get("reward")}')
else:
    print(f'✗ Failed: {r.text}')
