from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, Optional
from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask
import uvicorn

app = FastAPI(title="AI Social Guard OpenEnv")
env = AISocialGuardEnv()
current_task_id: Optional[str] = "easy"

# Store available tasks - Strictly easy, medium, hard
TASKS = {
    "easy": EasySpamTask,
    "medium": MediumReputationTask,
    "hard": HardGlobalModerationTask
}

@app.get("/")
def read_root():
    return {"status": "healthy", "env": "AISocialGuard"}

@app.post("/reset")
def reset(request: dict = Body(default={})):
    """
    Reset the environment for a specific task.
    Body: {"task_id": "easy", "seed": 42}
    """
    global current_task_id
    task_id = request.get("task_id", "easy")
    seed = request.get("seed", 42)
    
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")
    
    current_task_id = task_id
    task_cls = TASKS[task_id]
    task = task_cls()
    
    # AISocialGuardEnv.reset returns a SocialGuardObservation (Pydantic model)
    obs = env.reset(seed=seed, task_id=task_id, task_config=task.get_config())
    
    # FIX 2: Ensure serializable JSON return
    return {
        "observation": obs.model_dump(),
        "done": obs.done
    }

@app.post("/step")
def step(action: SocialGuardAction):
    """
    Take a step in the environment.
    Body: SocialGuardAction model
    """
    # env.step returns (obs, reward, done, info)
    obs, reward, done, info = env.step(action)
    
    # FIX 2: Ensure serializable JSON return
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
def get_state():
    """Returns the current raw state for grading."""
    return env.state().model_dump()

@app.post("/grade")
def get_grade():
    """Returns the official grade for the current task state."""
    global current_task_id
    if not current_task_id or current_task_id not in TASKS:
        raise HTTPException(status_code=400, detail="No active task to grade.")
    
    task_cls = TASKS[current_task_id]
    task = task_cls()
    score = task.grade(env.state())
    return {"task_id": current_task_id, "score": float(score)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
