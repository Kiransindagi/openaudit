from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from typing import Optional
from app.models import AuditAction, ResetResult
from app.env import get_env
import os

app = FastAPI(
    title="OpenAudit",
    version="1.0.0",
    description="AI Ecosystem Trust & Quality Auditing Environment",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {
        "service": "OpenAudit",
        "version": "1.0.0",
        "endpoints": {
            "reset": "POST /reset?task_id={task_id}",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

env = get_env()

@app.post("/reset")
def reset_episode(task_id: Optional[str] = Query(None)):
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
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": list(env.tasks.keys()),
        "count": len(env.tasks),
        "details": env.tasks
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "openaudit"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
