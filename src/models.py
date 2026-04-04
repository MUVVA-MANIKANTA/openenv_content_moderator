from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field

class ActionType(str, Enum):
    APPROVE = "approve"
    FLAG_SPAM = "flag_spam"
    FLAG_HATE = "flag_hate"
    FLAG_ADULT = "flag_adult"
    FLAG_MISINFO = "flag_misinfo"

class Post(BaseModel):
    post_id: int
    text: str
    user_id: int
    reputation: float  # 0.0 to 1.0
    # Internal/Internal-only field for simulation
    correct_label: Optional[ActionType] = Field(default=None, exclude=True)

class Action(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SocialGuardAction(Action):
    post_id: int
    action_type: ActionType
    reason: Optional[str] = Field(default=None, description="Optional reasoning for the action taken.")

class SocialGuardReward(BaseModel):
    value: float = Field(description="The numeric reward value for the action taken.")
    reason: str = Field(description="Brief explanation for the reward (e.g., 'Correct categorization', 'False Positive')")

class Observation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    # Adding a structured reward field for spec compliance if returned in metadata or info
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SocialGuardObservation(Observation):
    current_post: Optional[Dict[str, Any]] = None
    posts: List[Post]
    current_post_index: int
    total_posts: int
    message: Optional[str] = None

class State(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    episode_id: Optional[str] = None
    step_count: int = 0

class SocialGuardState(State):
    all_posts: List[Post]
    current_index: int = 0
    actions_taken: List[Dict[str, Any]] = []
    user_history: Dict[str, Dict[str, int]] = {}
