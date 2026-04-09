from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional
from app.models import AuditAction, ResetResult
from app.env import get_env
import os

app = FastAPI(title="OpenAudit", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>OpenAudit</h1>")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tasks")
def list_tasks():
    env = get_env()
    return {
        "tasks": list(env.tasks.keys()),
        "count": len(env.tasks),
        "details": env.tasks
    }

@app.post("/reset")
def reset_episode(task_id: Optional[str] = Query(None)):
    env = get_env()
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
def step_action(action: AuditAction):
    env = get_env()
    try:
        obs, reward, done, info = env.step(action)
        reward = round(min(0.99, max(0.01, float(reward))), 3)
        return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

@app.get("/state")
def get_state():
    env = get_env()
    return env.get_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


