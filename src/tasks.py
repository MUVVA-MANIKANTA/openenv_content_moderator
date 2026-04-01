import abc
from src.models import SocialGuardState, ActionType, Post
from typing import Dict, Any, List

class BaseTask(abc.ABC):
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def grade(self, state: SocialGuardState) -> float:
        pass

class EasySpamTask(BaseTask):
    """Detecting very obvious spam vs safe posts."""
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "text": "Win a brand new iPhone 16! Click here to claim your prize.", "user_id": 101, "reputation": 0.2, "correct_label": ActionType.FLAG_SPAM},
                {"post_id": 2, "text": "Hey check out this cool photo I took of the sunset.", "user_id": 102, "reputation": 0.8, "correct_label": ActionType.APPROVE},
                {"post_id": 3, "text": "URGENT: Your account will be suspended! Verify now at link.", "user_id": 103, "reputation": 0.1, "correct_label": ActionType.FLAG_SPAM},
                {"post_id": 4, "text": "Does anyone know how to fix a cracked screen?", "user_id": 104, "reputation": 0.5, "correct_label": ActionType.APPROVE},
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        total = len(state.all_posts)
        if total == 0: return 1.0
        correct = 0
        ground_truth = {p.post_id: p.correct_label for p in state.all_posts}
        for act in state.actions_taken:
            if act["action"] == ground_truth.get(act["post_id"]):
                correct += 1
        return correct / total

class MediumReputationTask(BaseTask):
    """Spam + Adult Content with user reputation influence."""
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "text": "BUY NFT NOW! BEST INVESTMENT EVER!", "user_id": 201, "reputation": 0.05, "correct_label": ActionType.FLAG_SPAM},
                {"post_id": 2, "text": "Adult site 18+ only, check it out.", "user_id": 202, "reputation": 0.3, "correct_label": ActionType.FLAG_ADULT},
                {"post_id": 3, "text": "My family had a great vacation this summer.", "user_id": 203, "reputation": 0.95, "correct_label": ActionType.APPROVE},
                {"post_id": 4, "text": "Looking for recommendations for a good thriller movie.", "user_id": 204, "reputation": 0.7, "correct_label": ActionType.APPROVE},
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        total = len(state.all_posts)
        if total == 0: return 0.0
        correct = 0
        ground_truth = {p.post_id: p.correct_label for p in state.all_posts}
        for act in state.actions_taken:
            if act["action"] == ground_truth.get(act["post_id"]):
                correct += 1
        return correct / total

class HardGlobalModerationTask(BaseTask):
    """All 4 categories + subtle misinformation + high-rep scrutiny."""
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "text": "Breaking news: New study shows eating plastic cures headaches.", "user_id": 301, "reputation": 0.1, "correct_label": ActionType.FLAG_MISINFO},
                {"post_id": 2, "text": "People who disagree with me are sub-human scum and don't belong here.", "user_id": 302, "reputation": 0.4, "correct_label": ActionType.FLAG_HATE},
                {"post_id": 3, "text": "This is a dangerous threat to our democracy. We must act now.", "user_id": 401, "reputation": 0.9, "correct_label": ActionType.APPROVE},
                {"post_id": 4, "text": "Join our secret group for exclusive adult content.", "user_id": 303, "reputation": 0.2, "correct_label": ActionType.FLAG_ADULT},
                {"post_id": 5, "text": "I heard that the recent election was rigged by lizard people.", "user_id": 402, "reputation": 0.7, "correct_label": ActionType.FLAG_MISINFO},
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        # Score 0.0-1.0 based on correct flags across all posts
        total = len(state.all_posts)
        if total == 0: return 0.0
        correct = 0
        ground_truth = {p.post_id: p.correct_label for p in state.all_posts}
        for act in state.actions_taken:
            # Special penalty for false misinfo flag on high-rep user
            if act["action"] == ActionType.FLAG_MISINFO and ground_truth.get(act["post_id"]) == ActionType.APPROVE:
                correct -= 0.5 # Severe penalty
            elif act["action"] == ground_truth.get(act["post_id"]):
                correct += 1
        return max(0.0, correct / total)

class CrisisResponseTask(BaseTask):
    """Ultra-hard: High-velocity misinformation attack from a serial offender."""
    def get_config(self) -> Dict[str, Any]:
        return {
            "posts": [
                {"post_id": 1, "text": "GOVERNMENT IS POISONING WATER! SHARE NOW!", "user_id": 999, "reputation": 0.01, "correct_label": ActionType.FLAG_MISINFO},
                {"post_id": 2, "text": "Please remain calm and follow official guidance.", "user_id": 100, "reputation": 0.9, "correct_label": ActionType.APPROVE},
                {"post_id": 3, "text": "THEY ARE LIARS! DON'T TRUST THE POLICE!", "user_id": 999, "reputation": 0.01, "correct_label": ActionType.FLAG_MISINFO},
                {"post_id": 4, "text": "Anyone who supports them is a traitor and should be exiled.", "user_id": 999, "reputation": 0.01, "correct_label": ActionType.FLAG_HATE},
                {"post_id": 5, "text": "Is the local clinic still open for emergencies?", "user_id": 101, "reputation": 0.6, "correct_label": ActionType.APPROVE},
            ]
        }

    def grade(self, state: SocialGuardState) -> float:
        total = len(state.all_posts)
        if total == 0: return 0.0
        correct = 0
        gt = {p.post_id: p.correct_label for p in state.all_posts}
        for act in state.actions_taken:
            # Special logic: Missing the second or third post from user 999 is a failure
            if act["post_id"] in [3, 4] and act["action"] != gt[act["post_id"]]:
                correct -= 1 # Heavy penalty for failing to stop a known offender
            elif act["action"] == gt[act["post_id"]]:
                correct += 1
        return max(0.0, correct / total)
class SampleDatasetTask(BaseTask):
    """Loads a custom dataset for evaluation."""
    def __init__(self, data_path: str = "data/training_samples.json"):
        import os
        import json
        self.data_path = data_path
        self.posts = []
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                self.posts = json.load(f)
        
    def get_config(self) -> Dict[str, Any]:
        # Convert string labels back to ActionType
        formatted_posts = []
        for p in self.posts:
            # Ensure action is ActionType
            label = p.get("correct_label", "approve").lower()
            try:
                action = ActionType(label)
            except ValueError:
                action = ActionType.APPROVE
                
            formatted_posts.append({
                "post_id": p.get("post_id"),
                "text": p.get("text"),
                "user_id": p.get("user_id"),
                "reputation": p.get("reputation"),
                "correct_label": action
            })
        return {"posts": formatted_posts}

    def grade(self, state: SocialGuardState) -> float:
        total = len(state.all_posts)
        if total == 0: return 0.0
        correct = 0
        ground_truth = {p.post_id: p.correct_label for p in state.all_posts}
        for act in state.actions_taken:
            if act["action"] == ground_truth.get(act["post_id"]):
                correct += 1
        return correct / total
