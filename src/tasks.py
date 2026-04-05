from typing import Dict, Any
from .models import SocialGuardState, ActionType

class BaseTask:
    def __init__(self):
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"posts": []}

    def grade(self, state: SocialGuardState) -> float:
        return 0.0

class EasySpamTask(BaseTask):
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "user_id": 101, "text": "Hello world, nice day!", "reputation": 0.9, "correct_label": "approve"},
                {"post_id": 2, "user_id": 102, "text": "Buy cheap watches at scam.com!", "reputation": 0.1, "correct_label": "flag_spam"},
                {"post_id": 3, "user_id": 103, "text": "Just ate an apple.", "reputation": 0.8, "correct_label": "approve"},
                {"post_id": 4, "user_id": 104, "text": "Click here to win a million dollars NOW!!!", "reputation": 0.05, "correct_label": "flag_spam"}
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        if not state.all_posts:
            return 0.0
            
        score = 0.0
        max_score = len(state.all_posts)
        
        gt = {p.post_id: ActionType(p.correct_label) for p in state.all_posts}
        
        for h in state.history:
            post_id = h.get("post_id")
            action = ActionType(h.get("action"))
            if post_id in gt:
                correct = gt[post_id]
                if action == correct:
                    score += 1.0
                elif action == ActionType.APPROVE and correct != ActionType.APPROVE:
                    score -= 0.5
        return max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0

class MediumReputationTask(BaseTask):
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "user_id": 201, "text": "Normal post.", "reputation": 0.9, "correct_label": "approve"},
                {"post_id": 2, "user_id": 202, "text": "Spam my link", "reputation": 0.1, "correct_label": "flag_spam"},
                {"post_id": 3, "user_id": 203, "text": "NSFW adult content here", "reputation": 0.2, "correct_label": "flag_adult"},
                {"post_id": 4, "user_id": 204, "text": "Vaccines have microchips to track you.", "reputation": 0.15, "correct_label": "flag_misinfo"},
                {"post_id": 5, "user_id": 205, "text": "Oh sure, the earth is flat. Sarcasm.", "reputation": 0.95, "correct_label": "approve"},
                {"post_id": 6, "user_id": 206, "text": "Drinking bleach cures all diseases.", "reputation": 0.01, "correct_label": "flag_misinfo"}
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        if not state.all_posts:
            return 0.0
        score = 0.0
        max_score = len(state.all_posts) * 2.0
        gt = {p.post_id: ActionType(p.correct_label) for p in state.all_posts}
        
        for h in state.history:
            post_id = h.get("post_id")
            action = ActionType(h.get("action"))
            if post_id in gt:
                correct = gt[post_id]
                if action == correct:
                    score += 2.0
                elif action == ActionType.APPROVE and correct in [ActionType.FLAG_MISINFO, ActionType.FLAG_HATE]:
                    score -= 0.5
                elif action != correct:
                    score += 0.5
        return max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0

class HardGlobalModerationTask(BaseTask):
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "user_id": 301, "text": "Peace.", "reputation": 0.99, "correct_label": "approve"},
                {"post_id": 2, "user_id": 302, "text": "I hate you all, literal scum.", "reputation": 0.1, "correct_label": "flag_hate"},
                {"post_id": 3, "user_id": 303, "text": "Look at my nude pics.", "reputation": 0.05, "correct_label": "flag_adult"},
                {"post_id": 4, "user_id": 304, "text": "Drink bleach as medicine.", "reputation": 0.05, "correct_label": "flag_misinfo"},
                {"post_id": 5, "user_id": 305, "text": "Buy crypto now!", "reputation": 0.1, "correct_label": "flag_spam"},
                {"post_id": 6, "user_id": 306, "text": "Satire: We should destroy the moon.", "reputation": 0.9, "correct_label": "approve"},
                {"post_id": 7, "user_id": 307, "text": "All people of group X are parasites and should be eliminated.", "reputation": 0.8, "correct_label": "flag_hate"},
                {"post_id": 8, "user_id": 308, "text": "I'm selling fake IDs.", "reputation": 0.2, "correct_label": "flag_spam"},
                {"post_id": 9, "user_id": 309, "text": "Election day is actually next week (false).", "reputation": 0.3, "correct_label": "flag_misinfo"}
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        if not state.all_posts:
            return 0.0
        score = 0.0
        max_score = len(state.all_posts) * 3.0
        gt = {p.post_id: ActionType(p.correct_label) for p in state.all_posts}
        
        for h in state.history:
            post_id = h.get("post_id")
            action = ActionType(h.get("action"))
            if post_id in gt:
                correct = gt[post_id]
                if action == correct:
                    score += 3.0
                elif action == ActionType.APPROVE and correct != ActionType.APPROVE:
                    score -= 0.5
                elif action != ActionType.APPROVE and correct != ActionType.APPROVE:
                    score += 1.0
        return max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0

class CrisisResponseTask(BaseTask):
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "user_id": 999, "text": "The hospital was just attacked!", "reputation": 0.1, "correct_label": "flag_misinfo"},
                {"post_id": 2, "user_id": 501, "text": "Are you sure? I heard it's fine.", "reputation": 0.8, "correct_label": "approve"},
                {"post_id": 3, "user_id": 999, "text": "Yes, attacked! 50 dead! Spread the word!", "reputation": 0.1, "correct_label": "flag_misinfo"},
                {"post_id": 4, "user_id": 999, "text": "The army is coming for us next! Arm yourselves!", "reputation": 0.1, "correct_label": "flag_misinfo"},
                {"post_id": 5, "user_id": 502, "text": "Stop spreading rumors, it was just a fire alarm.", "reputation": 0.9, "correct_label": "approve"}
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        if not state.all_posts:
            return 0.0
        score = 0.0
        max_score = len(state.all_posts) * 2.0
        gt = {p.post_id: ActionType(p.correct_label) for p in state.all_posts}
        
        history_actions = {h.get("post_id"): ActionType(h.get("action")) for h in state.history}
        
        for post_id, correct in gt.items():
            if post_id in history_actions:
                action = history_actions[post_id]
                if action == correct:
                    score += 2.0
                elif action == ActionType.APPROVE and correct != ActionType.APPROVE:
                    score -= 0.5
                elif action != ActionType.APPROVE and correct != ActionType.APPROVE:
                    score += 1.0
                    
        # Penalty for missing serial offender escalations
        if history_actions.get(3) == ActionType.APPROVE:
            score -= 1.0
        if history_actions.get(4) == ActionType.APPROVE:
            score -= 1.0
            
        return max(0.0, min(1.0, score / max_score)) if max_score > 0 else 0.0
