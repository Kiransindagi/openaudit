from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from app.models import AuditAction, ResetResult
from app.env import get_env

# Create the FastAPI app instance
app = FastAPI(title="OpenAudit", version="1.0.0", description="AI Ecosystem Trust & Quality Auditing Environment")

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "service": "OpenAudit",
        "version": "1.0.0",
        "description": "AI Ecosystem Trust & Quality Auditing Environment",
        "endpoints": {
            "reset": "POST /reset?task_id={task_id}",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "space_url": "https://kiransin-openaudit.hf.space"
    }

# Get global environment instance
env = get_env()

@app.post("/reset")
def reset_episode(task_id: Optional[str] = Query(None, description="Task ID to run")):
    """Reset the audit state machine for a new episode."""
    try:
        observation = env.reset(task_id)
        return ResetResult(
            observation=observation,
            info={
                "pillar": observation.artifact_type,
                "difficulty": task_id.split("_")[-1] if task_id else "easy",
                "max_steps": observation.max_steps,
                "total_flaws": observation.total_flaws
            }
        ).dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/step")
def step(action: AuditAction):
    """Execute one step in the audit state machine."""
    try:
        observation, total_reward, done, info = env.step(action)
        return {
            "observation": observation.dict(),
            "reward": total_reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

@app.get("/state")
def get_state():
    """Get current state machine status."""
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")

@app.get("/tasks")
def get_tasks():
    """List all available tasks."""
    return {
        "tasks": list(env.tasks.keys()),
        "count": len(env.tasks),
        "details": env.tasks
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "openaudit"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
