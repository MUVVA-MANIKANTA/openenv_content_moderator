from typing import Tuple, Dict, Any
from .models import (
    SocialGuardObservation, SocialGuardAction, SocialGuardReward,
    SocialGuardState, ActionType, Post
)

class AISocialGuardEnv:
    def __init__(self):
        self._state = SocialGuardState()
        self._seed = 42

    def reset(self, seed: int = 42, task_config: Dict[str, Any] = None) -> SocialGuardObservation:
        self._seed = seed
        self._state = SocialGuardState()
        
        if task_config and "posts" in task_config:
            self._state.all_posts = [Post(**p) for p in task_config["posts"]]
        
        return self._get_observation()

    def _get_observation(self) -> SocialGuardObservation:
        current_post = None
        message = None
        
        if self._state.current_index < len(self._state.all_posts):
            post = self._state.all_posts[self._state.current_index]
            current_post = {
                "post_id": post.post_id,
                "user_id": post.user_id,
                "text": post.text,
                "reputation": post.reputation
            }
        else:
            message = "All posts moderated."

        return SocialGuardObservation(
            current_post=current_post,
            posts=self._state.all_posts,
            current_post_index=self._state.current_index,
            total_posts=len(self._state.all_posts),
            message=message
        )

    def compute_reward(self, action: SocialGuardAction, post: Post) -> SocialGuardReward:
        raw_reward = 0.0
        reason_parts = []
        
        correct_label = ActionType(post.correct_label)
        
        if action.action_type == correct_label:
            if correct_label == ActionType.APPROVE:
                raw_reward += 0.1
                reason_parts.append("Correctly approved.")
            else:
                raw_reward += 0.2
                reason_parts.append("Correctly flagged.")
                if post.reputation < 0.2:
                    raw_reward += 0.1
                    reason_parts.append("Low-rep offender caught (+0.1 bonus).")
        else:
            if correct_label == ActionType.APPROVE:
                if post.reputation > 0.8:
                    raw_reward -= 0.2
                    reason_parts.append("False positive on trusted user (-0.2).")
                else:
                    raw_reward -= 0.1
                    reason_parts.append("False positive (-0.1).")
            else:
                if action.action_type == ActionType.APPROVE:
                    raw_reward -= 0.2
                    reason_parts.append("Missed toxic post (-0.2).")
                else:
                    raw_reward -= 0.1
                    reason_parts.append(f"Wrong flag category (was {correct_label.value}) (-0.1).")

        user_key = str(post.user_id)
        if user_key not in self._state.user_history:
            self._state.user_history[user_key] = {"misses": 0}

        # Serial offender escalating penalty ONLY when current action is WRONG
        # AND it genuinely was a bad post (not an approve)
        if action.action_type != correct_label and correct_label != ActionType.APPROVE:
            miss_count = self._state.user_history[user_key]["misses"]
            penalty = min(miss_count * 0.15, 0.45)
            raw_reward -= penalty
            if penalty > 0:
                reason_parts.append(f"Serial offender penalty (-{penalty:.2f}).")
            self._state.user_history[user_key]["misses"] += 1

        if action.reason:
            raw_reward += 0.01

        _RAW_MIN, _RAW_MAX = -0.8, 0.4
        normalized = (raw_reward - _RAW_MIN) / (_RAW_MAX - _RAW_MIN)
        final_value = round(max(0.0, min(1.0, normalized)), 4)
        
        return SocialGuardReward(value=final_value, reason=" ".join(reason_parts) or "No reason.")

    def step(self, action: SocialGuardAction) -> Tuple[SocialGuardObservation, SocialGuardReward, bool, Dict[str, Any]]:
        info = {}
        if self._state.current_index >= len(self._state.all_posts):
            return self._get_observation(), SocialGuardReward(value=0.0, reason="Episode done"), True, info
            
        current_post = self._state.all_posts[self._state.current_index]
        if current_post.post_id != action.post_id:
            reward = SocialGuardReward(value=0.0, reason=f"Action post_id does not match")
            return self._get_observation(), reward, False, info

        reward = self.compute_reward(action, current_post)
        
        self._state.history.append({
            "post_id": current_post.post_id,
            "action": action.action_type.value,
            "reward": reward.model_dump()
        })
        
        self._state.current_index += 1
        done = self._state.current_index >= len(self._state.all_posts)
        
        return self._get_observation(), reward, done, info

    def state(self) -> SocialGuardState:
        return self._state
