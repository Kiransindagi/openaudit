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
