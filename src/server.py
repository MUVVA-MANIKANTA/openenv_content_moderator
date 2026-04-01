from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
from .env import AISocialGuardEnv
from .models import SocialGuardAction, SocialGuardObservation, SocialGuardState, ActionType
from .tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask
from .ui import create_gradio_demo
import gradio as gr

app = FastAPI(title="AI Social Guard OpenEnv")
env = AISocialGuardEnv()

# Mount Gradio UI for Demo Visibility
demo = create_gradio_demo()
app = gr.mount_gradio_app(app, demo, path="/ui")

# Store available tasks
TASKS = {
    "easy": EasySpamTask,
    "medium": MediumReputationTask,
    "hard": HardGlobalModerationTask,
    "easy_spam": EasySpamTask,
    "medium_reputation": MediumReputationTask,
    "hard_global": HardGlobalModerationTask,
    "crisis_response": CrisisResponseTask
}

@app.get("/")
def read_root():
    return {"status": "healthy", "env": "AISocialGuard"}

@app.post("/reset")
def reset(seed: Optional[int] = None, task_id: Optional[str] = "easy_spam"):
    task_cls = TASKS.get(task_id, EasySpamTask)
    task = task_cls()
    obs = env.reset(seed=seed, task_config=task.get_config())
    # Convert to dict for response
    return {"observation": obs.model_dump(), "done": obs.done}

@app.post("/step")
def step(action: SocialGuardAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "done": done,
        "reward": reward,
        "info": info
    }

@app.get("/state")
def get_state():
    return env.state().model_dump()

@app.get("/schema")
def get_schema():
    return {
        "action": SocialGuardAction.model_json_schema(),
        "observation": SocialGuardObservation.model_json_schema(),
        "state": SocialGuardState.model_json_schema()
    }
