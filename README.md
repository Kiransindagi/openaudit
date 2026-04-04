---
title: OpenAudit
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags: [openenv]
---

# OpenAudit

AI ecosystem trust & quality auditing environment for Hugging Face.

## Endpoints

- `POST /reset` - Start new audit episode
- `POST /step` - Submit audit action  
- `GET /state` - Get current state

## Environment

OpenEnv-compatible environment with 12 tasks across 4 pillars:
- Model Card Auditing
- Dataset Quality Control
- RL Reward Verification
- Tool Safety Testing

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

The observation includes `findings_so_far` to help you avoid repeats.

<!-- force rebuild -->
