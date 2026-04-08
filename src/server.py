from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
from .env import AISocialGuardEnv
from .models import SocialGuardAction, ActionType
from .tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask
from .ui import create_gradio_demo
import gradio as gr

class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    task_id: Optional[str] = "easy_spam"
    task_config: Optional[Dict[str, Any]] = None

TASKS = {
    "easy_spam":         EasySpamTask,
    "medium_reputation": MediumReputationTask,
    "hard_global":       HardGlobalModerationTask,
    "crisis_response":   CrisisResponseTask,
    "easy":              EasySpamTask,
    "medium":            MediumReputationTask,
    "hard":              HardGlobalModerationTask,
}

app = FastAPI(title="AI Social Guard")
env = AISocialGuardEnv()
current_task = None

@app.get("/")
def read_root():
    return {"status": "healthy", "env": "AISocialGuard", "version": "2.1.0"}

@app.post("/reset")
def reset(body: Optional[ResetRequest] = Body(None)):
    global current_task
    req = body or ResetRequest()
    task_class = TASKS.get(req.task_id, EasySpamTask)
    current_task = task_class()
    
    config = req.task_config or current_task.get_config()
    obs = env.reset(seed=req.seed, task_config=config)
    
    return {"observation": obs.model_dump(), "done": obs.total_posts == 0}

@app.post("/step")
def step(action: SocialGuardAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state().model_dump()

demo = create_gradio_demo()
app = gr.mount_gradio_app(app, demo, path="/")
