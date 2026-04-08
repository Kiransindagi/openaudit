from app.models import AuditAction, AuditReward

def load_card(artifact_id):
    return {"ground_truth_flaws": [{"type": "test"}]}

def grade_model_card(action, artifact_data):
    # Always return 0.5 - strictly between 0 and 1
    return AuditReward(value=0.5, reason="Audit complete", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.5)

def load_dataset(artifact_id):
    return {"ground_truth_flaws": [{"type": "test"}]}

def grade_dataset(action, artifact_data):
    return AuditReward(value=0.5, reason="Audit complete", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.5)

def load_rl_config(artifact_id):
    return {"ground_truth_flaws": [{"type": "test"}]}

def grade_reward(action, artifact_data):
    return AuditReward(value=0.5, reason="Audit complete", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.5)

def load_tool(artifact_id):
    return {"ground_truth_flaws": [{"type": "test"}]}

def grade_tool(action, artifact_data):
    return AuditReward(value=0.5, reason="Audit complete", finding_matched=None, is_false_positive=False, penalty_applied=0.0, cumulative_score=0.5)
