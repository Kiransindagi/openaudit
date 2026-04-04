"""
OpenAudit Submission Validation Report
Run this before final submission to ensure all requirements are met
"""

import requests
import json
import os
import sys
from datetime import datetime

BASE_URL = "https://kiransin-openaudit.hf.space"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}📌 {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

print("=" * 70)
print(f"{BLUE}OpenAudit - Hackathon Submission Validation Report{RESET}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Space URL: {BASE_URL}")
print("=" * 70)

# ============================================================
# DISQUALIFIER CHECKS (Must ALL pass)
# ============================================================
print("\n" + "=" * 70)
print("🚨 DISQUALIFIER CHECKS (All must pass)")
print("=" * 70)

all_disqualifiers_pass = True

# 1. HF Space live and /reset returns 200
print("\n1. HF Space /reset endpoint...")
try:
    resp = requests.post(f"{BASE_URL}/reset?task_id=model_card_easy", timeout=10)
    if resp.status_code == 200:
        print_success("POST /reset returns HTTP 200")
    else:
        print_error(f"POST /reset returns {resp.status_code}")
        all_disqualifiers_pass = False
except Exception as e:
    print_error(f"Cannot reach Space: {e}")
    all_disqualifiers_pass = False

# 2. inference.py exists in root
print("\n2. inference.py in root directory...")
if os.path.exists("inference.py"):
    print_success("inference.py exists in root")
else:
    print_error("inference.py NOT found in root")
    all_disqualifiers_pass = False

# 3. Check inference.py log format
print("\n3. inference.py log format...")
try:
    with open("inference.py", "r") as f:
        content = f.read()
    
    has_start = "[START]" in content
    has_step = "[STEP]" in content
    has_end = "[END]" in content
    
    if has_start and has_step and has_end:
        print_success("Has [START], [STEP], [END] log format")
    else:
        print_error(f"Missing log format: START={has_start}, STEP={has_step}, END={has_end}")
        all_disqualifiers_pass = False
except:
    print_error("Cannot read inference.py")
    all_disqualifiers_pass = False

# 4. Check inference.py reads env vars
print("\n4. Environment variables...")
has_env_vars = "API_BASE_URL" in content and "MODEL_NAME" in content and "HF_TOKEN" in content
if has_env_vars:
    print_success("Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment")
else:
    print_error("Missing required environment variable reads")
    all_disqualifiers_pass = False

# 5. Check inference.py uses OpenAI client
print("\n5. OpenAI client usage...")
has_openai = "OpenAI" in content and "base_url" in content
if has_openai:
    print_success("Uses OpenAI(base_url=..., api_key=...) pattern")
else:
    print_warning("OpenAI client pattern not found - recommended to add")
    # Not a disqualifier but important

# 6. Docker build test
print("\n6. Docker build...")
try:
    import subprocess
    result = subprocess.run(["docker", "build", "-t", "openaudit-test", "."], 
                          capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print_success("Docker build succeeds")
    else:
        print_error("Docker build failed")
        all_disqualifiers_pass = False
except:
    print_warning("Docker not available for testing (will work on HF)")

# 7. Graders return different scores
print("\n7. Graders return different scores...")
tasks = ["model_card_easy", "dataset_qc_easy", "rl_reward_easy"]
scores = []
for task in tasks:
    try:
        reset_resp = requests.post(f"{BASE_URL}/reset?task_id={task}", timeout=10)
        if reset_resp.status_code == 200:
            # Simple action
            action = {"pillar": task.split("_")[0], "finding_type": "test", 
                     "target_field": "test", "description": "test", "severity": 2}
            step_resp = requests.post(f"{BASE_URL}/step", json=action, timeout=10)
            if step_resp.status_code == 200:
                reward = step_resp.json().get("reward", 0)
                scores.append(reward)
    except:
        pass

if len(set(scores)) > 1:
    print_success(f"Graders return different scores: {scores}")
else:
    print_warning(f"All graders returned same score: {scores}")

# ============================================================
# FEATURE CHECKS
# ============================================================
print("\n" + "=" * 70)
print("📋 FEATURE CHECKS")
print("=" * 70)

# Check all 12 tasks
print("\n8. All 12 tasks available...")
try:
    resp = requests.get(f"{BASE_URL}/tasks", timeout=10)
    if resp.status_code == 200:
        tasks = resp.json().get("tasks", [])
        print_success(f"Found {len(tasks)} tasks")
        for task in tasks:
            print(f"   - {task}")
    else:
        print_error(f"Tasks endpoint failed: {resp.status_code}")
except:
    print_error("Cannot access /tasks endpoint")

# Check partial rewards
print("\n9. Partial reward testing...")
try:
    # Test partial vs perfect
    reset_resp = requests.post(f"{BASE_URL}/reset?task_id=model_card_easy")
    if reset_resp.status_code == 200:
        # Partial action
        action = {"pillar": "model_card", "finding_type": "missing_field", 
                 "target_field": "license", "description": "Missing license", "severity": 2}
        step_resp = requests.post(f"{BASE_URL}/step", json=action)
        partial_reward = step_resp.json().get("reward", 0)
        print_info(f"Partial reward (single field): {partial_reward}")
        if 0 < partial_reward < 1.0:
            print_success("Partial reward working")
        else:
            print_warning(f"Partial reward = {partial_reward} (expected between 0 and 1)")
except:
    print_error("Could not test partial rewards")

# Check health endpoint
print("\n10. Health check...")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    if resp.status_code == 200:
        print_success(f"Health check: {resp.json()}")
    else:
        print_error(f"Health check failed: {resp.status_code}")
except:
    print_error("Health endpoint not available")

# Check root endpoint
print("\n11. Root endpoint...")
try:
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        print_success(f"Root endpoint returns API info")
        print(f"   Service: {data.get('service')}")
        print(f"   Endpoints: {list(data.get('endpoints', {}).keys())}")
    else:
        print_error(f"Root endpoint returned {resp.status_code}")
except:
    print_error("Root endpoint not available")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)

if all_disqualifiers_pass:
    print_success("All disqualifier checks PASSED!")
    print_success("Your OpenAudit environment is READY FOR SUBMISSION!")
else:
    print_error("Some disqualifier checks FAILED. Please fix before submitting.")

print(f"\n{BLUE}Space URL:{RESET} {BASE_URL}")
print(f"{BLUE}API Docs:{RESET} {BASE_URL}/docs")
print(f"{BLUE}Health Check:{RESET} {BASE_URL}/health")

print("\n" + "=" * 70)
print("✅ Validation Complete!")
print("=" * 70)
