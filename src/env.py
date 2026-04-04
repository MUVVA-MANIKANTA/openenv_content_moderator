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

    def step(self, action: SocialGuardAction) -> tuple[SocialGuardObservation, SocialGuardReward, bool, dict]:
        if self._state.current_index >= len(self._state.all_posts):
            obs = self._get_observation(message="No more posts to moderate.")
            return obs, SocialGuardReward(value=0.0, reason="No posts"), True, {}

        post = self._state.all_posts[self._state.current_index]
        reward_score = self.compute_reward(post, action)
        
        # Normalization: map raw reward [-0.5, 0.4] to [0.0, 1.0]
        _RAW_MIN, _RAW_MAX = -0.5, 0.4
        normalized = (reward_score - _RAW_MIN) / (_RAW_MAX - _RAW_MIN)
        final_reward_value = round(max(0.0, min(1.0, normalized)), 4)
        
        reason_str = "Correct action taken" if action.action_type == post.correct_label else "Incorrect action taken"

        # If the post WAS toxic and the agent DIDN'T catch it, record a 'miss'.
        was_missed = (post.correct_label != ActionType.APPROVE and action.action_type != post.correct_label)
        user_id = str(post.user_id)
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
        obs.reward = final_reward_value
        obs.done = done
        
        # Info dictionary with reasoning quality feedback and threat level
        info = {
            "threat_level": min(1.0, self._state.user_history[user_id]["missed_violations"] * 0.3),
            "user_stats": self._state.user_history[user_id]
        }
        
        final_reward = SocialGuardReward(value=final_reward_value, reason=reason_str)
        return obs, final_reward, done, info

    def compute_reward(self, post: Post, action: SocialGuardAction) -> float:
        correct_label = post.correct_label
        base_reward = 0.0
        
        # 1. Base Label Logic
        if action.action_type == correct_label:
            if action.action_type == ActionType.APPROVE:
                base_reward += 0.1 # Correct approve
            else:
                base_reward += 0.2 # Correct flag
                if post.reputation < 0.2:
                    base_reward += 0.1 # Bonus for catching low-rep offender
        else:
            if action.action_type != ActionType.APPROVE and correct_label == ActionType.APPROVE:
                penalty = 0.2 if post.reputation > 0.8 else 0.1 # Reputation penalty on false positive
                base_reward -= penalty
            elif action.action_type == ActionType.APPROVE and correct_label != ActionType.APPROVE:
                base_reward -= 0.2 # Wrong action missing toxic
            else:
                base_reward -= 0.1 # General wrong action
                
        # Serial offender escalating penalty (miss_count * 0.15)
        user_id = str(post.user_id)
        if user_id in self._state.user_history:
            miss_count = self._state.user_history[user_id].get("missed_violations", 0)
            if miss_count > 0:
                base_reward -= (miss_count * 0.15)

        # 2. Reasoning Bonus +0.01 if reason provided
        if action.reason and len(action.reason) > 0:
            base_reward += 0.01

        return float(base_reward)

    def _get_observation(self, message: Optional[str] = None) -> SocialGuardObservation:
        current_p = None
        if self._state.current_index < len(self._state.all_posts):
            p = self._state.all_posts[self._state.current_index]
            current_p = {"text": p.text, "reputation": p.reputation, "post_id": p.post_id, "user_id": p.user_id}

        return SocialGuardObservation(
            current_post=current_p,
            posts=self._state.all_posts[self._state.current_index :],
            current_post_index=self._state.current_index,
            total_posts=len(self._state.all_posts),
            reward=0.0, # Initial/Default reward for new observation
            done=self._state.current_index >= len(self._state.all_posts),
            message=message
        )
