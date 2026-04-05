from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    APPROVE = "approve"
    FLAG_SPAM = "flag_spam"
    FLAG_HATE = "flag_hate"
    FLAG_ADULT = "flag_adult"
    FLAG_MISINFO = "flag_misinfo"

class Post(BaseModel):
    post_id: int
    user_id: int
    text: str
    reputation: float
    correct_label: str = Field(exclude=True)

class Observation(BaseModel):
    pass

class Action(BaseModel):
    pass

class SocialGuardObservation(Observation):
    current_post: Optional[Dict[str, Any]] = None
    posts: List[Post] = []
    current_post_index: int = 0
    total_posts: int = 0
    message: Optional[str] = None

class SocialGuardAction(Action):
    post_id: int
    action_type: ActionType
    reason: Optional[str] = None

class SocialGuardReward(BaseModel):
    value: float
    reason: str

class SocialGuardState(BaseModel):
    all_posts: List[Post] = []
    current_index: int = 0
    history: List[Dict[str, Any]] = []
    user_history: Dict[str, Dict[str, int]] = {}
