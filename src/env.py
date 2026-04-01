from typing import List, Dict, Any, Optional
import uuid
import random
from src.models import SocialGuardAction, SocialGuardObservation, SocialGuardState, Post, ActionType, SocialGuardReward

class AISocialGuardEnv:
    def __init__(self, task_config: Dict[str, Any] = None):
        self._state = SocialGuardState(all_posts=[], current_index=0)
        self.task_config = task_config or {}

    def state(self) -> SocialGuardState:
        """Returns the current state of the environment."""
        return self._state

    def reset(self, seed: Optional[int] = 42, task_id: Optional[str] = "easy_spam", task_config: Dict[str, Any] = None, **kwargs) -> SocialGuardObservation:
        # Enforce determinism for graders
        random.seed(seed)
        
        # Load posts from task_config (fallback to self.task_config)
        config = task_config or self.task_config
        posts_data = config.get("posts", []) if config else []
        all_posts = [Post(**p) for p in posts_data]
            
        self._state = SocialGuardState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            all_posts=all_posts,
            current_index=0,
            actions_taken=[],
            user_history={} # New: Tracks history of violations per user_id
        )
        
        return self._get_observation(message="AI Social Guard Environment reset.")

    def step(self, action: SocialGuardAction) -> tuple[SocialGuardObservation, float, bool, dict]:
        if self._state.current_index >= len(self._state.all_posts):
            obs = self._get_observation(message="No more posts to moderate.")
            return obs, 0.0, True, {}

        post = self._state.all_posts[self._state.current_index]
        reward_score = self.compute_reward(post, action)
        
        # OpenEnv Mandatory: Score must be between 0.0 and 1.0
        reward_score = max(0.0, min(1.0, reward_score))
        # If the post WAS toxic and the agent DIDN'T catch it, record a 'miss'.
        was_missed = (post.correct_label != ActionType.APPROVE and action.action_type != post.correct_label)
        user_id = post.user_id
        if user_id not in self._state.user_history:
            self._state.user_history[user_id] = {"missed_violations": 0, "total_posts": 0}
        
        self._state.user_history[user_id]["total_posts"] += 1
        if was_missed:
            self._state.user_history[user_id]["missed_violations"] += 1

        # Track action taken
        self._state.actions_taken.append({
            "post_id": post.post_id,
            "action": action.action_type,
            "reward": reward_score,
            "reason": action.reason
        })
        
        # Advance index
        self._state.current_index += 1
        self._state.step_count += 1
        
        done = self._state.current_index >= len(self._state.all_posts)
        
        obs = self._get_observation(message=f"Action '{action.action_type}' applied.")
        obs.reward = reward_score
        obs.done = done
        
        # Info dictionary with reasoning quality feedback and threat level
        info = {
            "threat_level": min(1.0, self._state.user_history[user_id]["missed_violations"] * 0.3),
            "user_stats": self._state.user_history[user_id]
        }
        
        return obs, float(reward_score), done, info

    def compute_reward(self, post: Post, action: SocialGuardAction) -> float:
        correct_label = post.correct_label
        
        # 1. Base Logic (Already robust)
        reputation_penalty = 0.0
        if action.action_type != ActionType.APPROVE and correct_label == ActionType.APPROVE:
            reputation_penalty = 0.1 * post.reputation 
            
        reputation_bonus = 0.0
        if action.action_type in [ActionType.FLAG_MISINFO, ActionType.FLAG_HATE] and correct_label == action.action_type:
            reputation_bonus = 0.1 * (1.0 - post.reputation) 

        base_reward = 0.0
        if action.action_type == correct_label:
            base_reward = 0.2 if action.action_type != ActionType.APPROVE else 0.1
            base_reward += reputation_bonus
        else:
            base_reward = -0.1 if action.action_type != ActionType.APPROVE else -0.2
            base_reward -= reputation_penalty
            
        # 2. Advanced: THE SERIAL OFFENDER PENALTY
        # If the user has missed violations in the past, a new miss is penalized more.
        user_stats = self._state.user_history.get(post.user_id, {"missed_violations": 0})
        if action.action_type != correct_label and correct_label != ActionType.APPROVE:
            # Escalating penalty for failing to catch repeat offenders
            miss_count = user_stats["missed_violations"]
            if miss_count >= 1:
                base_reward -= 0.15 * miss_count # Escalating penalty for repeat misses
                
        # 3. Reasoning Bonus: Reward agent for providing justification (minimal for now)
        if action.reason and len(action.reason) > 5:
            base_reward += 0.01

        return round(float(base_reward), 3)

    def _get_observation(self, message: Optional[str] = None) -> SocialGuardObservation:
        current_p = None
        if self._state.current_index < len(self._state.all_posts):
            p = self._state.all_posts[self._state.current_index]
            current_p = {"text": p.text, "reputation": p.reputation, "post_id": p.post_id}

        return SocialGuardObservation(
            current_post=current_p,
            posts=self._state.all_posts[self._state.current_index :],
            current_post_index=self._state.current_index,
            total_posts=len(self._state.all_posts),
            reward=0.0, # Initial/Default reward for new observation
            done=self._state.current_index >= len(self._state.all_posts),
            message=message
        )
