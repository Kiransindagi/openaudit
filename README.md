---
title: OpenAudit
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0.0"
pinned: false
tags:
  - openenv
  - auditing
  - trust
  - safety
  - hackathon
app_port: 7860
---

# OpenAudit: AI Ecosystem Trust & Quality Auditing

## What is OpenAudit?

OpenAudit is a production-grade **OpenEnv environment** for training AI agents to audit Hugging Face ecosystem artifacts. With 2M+ models and 500K+ datasets, manual quality verification is impossible. OpenAudit trains agents to automatically find missing fields, license conflicts, benchmark fraud, data quality issues, reward hacking, and dangerous tool code.

## 13 Tasks Across 4 Pillars

| Task ID | Pillar | Difficulty | Objective | Grader Logic |
|---------|--------|------------|-----------|--------------|
| model_card_easy | Model Card | Easy | Find missing fields | found_missing / total_missing |
| model_card_medium | Model Card | Medium | Detect license conflicts | binary + partial for naming parent |
| model_card_hard | Model Card | Hard | Verify benchmark claims | match_found(0.5) + numbers(0.5) |
| dataset_qc_easy | Dataset QC | Easy | Detect null values | columns_correct / total_columns |
| dataset_qc_medium | Dataset QC | Medium | Find duplicates | F1 on duplicate pairs |
| dataset_qc_hard | Dataset QC | Hard | Identify test leakage | precision × recall |
| rl_reward_easy | RL Reward | Easy | Flag sparse rewards | binary(0.6) + ratio(0.4) |
| rl_reward_medium | RL Reward | Medium | Detect reward hacking | keyword match |
| rl_reward_hard | RL Reward | Hard | Propose verifier fix | test_cases_passing / 5 |
| tool_tester_easy | Tool Tester | Easy | Code quality issues | issues_found / 3 |
| tool_tester_medium | Tool Tester | Medium | Silent failure detection | both_issues_found (binary) |
| tool_tester_hard | Tool Tester | Hard | Adversarial chain | correct_tool(0.5) + reason(0.5) |
| **model_card_audit_chain** | **Model Card** | **Hard** | **Multi-step audit** | **phase-based rewards** |

## Multi-Step Audit Chain

The `model_card_audit_chain` task requires three phases:

| Phase | Action | Reward | Description |
|-------|--------|--------|-------------|
| **SCAN** | `scan` | 0.2-0.3 | Identify potential issues |
| **INVESTIGATE** | `investigate` | 0.4-0.6 | Provide specific evidence |
| **REPORT** | `report` | 0.8-1.0 | Deliver final findings |

## Reward Design

- **Partial credit**: Each correct finding adds 0.2-1.2 points depending on completeness
- **False positive penalty**: -0.2 for incorrect findings
- **Step penalty**: -0.05 for inefficient exploration
- **Completion bonus**: +0.2 for finding all flaws early

## Baseline Scores (from LLM-based inference)
model_card_easy: reward=0.53, score=0.107
dataset_qc_easy: reward=1.10, score=0.220
rl_reward_easy: reward=1.00, score=0.200
Overall score: 0.176

text

## Synthetic Artifacts

All artifacts are generated deterministically with known flaws:

### Model Cards (10 files)
- `card_0.json` (Easy): Missing license, evaluation results, CO2 emissions
- `card_1.json` (Medium): MIT license conflict with GPL-3.0 parent model
- `card_2.json` (Hard): MMLU benchmark fraud (claimed 87.3 vs actual 81.2)
- Cards 3-9: Variations of the above with random flaws

### Datasets (10 files)
- `dataset_0.json` (Easy): 8 null values across columns colA, colB, colC
- `dataset_1.json` (Medium): 5 exact duplicates + 3 near-duplicates
- `dataset_2.json` (Hard): 4 test samples leaked into train split
- Each dataset has 100 rows with deterministic splits

### RL Configs (10 files)
- `rl_0.json` (Easy): Sparse reward (9 zeros, 1 one)
- `rl_1.json` (Medium): Reward hacking via "YES" token
- `rl_2.json` (Hard): Broken verifier always returns 1.0

### Tools (10 files)
- `tool_0.json` (Easy): Function with no docstring, type hints, or return annotation
- `tool_1.json` (Medium): Swallows exceptions and returns None
- `tool_2.json` (Hard): Adversarial chain – Tool C executes arbitrary code

## Action-to-Environment Mapping

When you submit an action via `/step`, the environment:

1. Validates the `pillar` matches the current task (else penalty -0.2)
2. Routes to the appropriate grader based on `finding_type`
3. Compares your `description` against ground truth keywords
4. Returns a reward:
   - **0.2–0.8** for partial matches
   - **0.9–1.2** for perfect matches
   - **-0.2** for false positives
5. Updates `flaws_found_count` and checks if all flaws are discovered
6. Ends episode when all flaws found or `max_steps` reached

## Setup & Usage

### Prerequisites
- Docker (recommended) or Python 3.11+
- Hugging Face account with token

### Run with Docker

```bash
docker build -t openaudit .
docker run -p 7860:7860 openaudit
Run with Python
bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
Test the API
bash
# Reset episode
curl -X POST "http://localhost:7860/reset?task_id=model_card_easy"

# Submit a finding
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"pillar":"model_card","finding_type":"missing_field","target_field":"license","description":"Missing license","severity":2}'

# Check state
curl -X GET "http://localhost:7860/state"
Run the Baseline Agent
bash
export ENV_API_URL="https://kiransin-openaudit.hf.space"
export LLM_API_BASE="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
python inference.py
API Endpoints
EndpointMethodDescription
/GETAPI information
/healthGETHealth check
/tasksGETList all 13 tasks
/resetPOSTStart new episode
/stepPOSTSubmit action
/stateGETCurrent state
/docsGETSwagger UI
Live Space
https://kiransin-openaudit.hf.space

License
MIT
