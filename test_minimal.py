import os
import sys
import random

# Ensure root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import AISocialGuardEnv
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask
from src.models import SocialGuardAction, ActionType

def run_minimal_test():
    # Use actual class instances from tasks.py
    tasks = [EasySpamTask(), MediumReputationTask(), HardGlobalModerationTask(), CrisisResponseTask()]
    print("="*40)
    print("   AI SOCIAL GUARD: MINIMAL TEST RUN   ")
    print("="*40)
    
    for task in tasks:
        print(f"\n--- MISSION: {task.__class__.__name__} ---")
        rng = random.Random(42)
        env = AISocialGuardEnv(task_config=task.get_config())
        obs = env.reset(seed=42)
        done = False
        
        while not done:
            if not obs.posts: break
            
            current_post = obs.current_post
            if not current_post: break
            
            gt = {p["post_id"]: ActionType(p["correct_label"]) for p in task.get_config()["posts"]}
            correct_label = gt.get(current_post["post_id"], ActionType.APPROVE)
            
            # 70% chance to be a 'smart' agent, 30% random
            if rng.random() < 0.7:
                action_type = correct_label
            else:
                action_type = rng.choice(list(ActionType))
                
            action = SocialGuardAction(
                post_id=current_post["post_id"],
                action_type=action_type
            )
            obs, reward, done, info = env.step(action)
            
            reward_val = reward.value if hasattr(reward, 'value') else reward
            print(f"  [STEP] Post {current_post['post_id']}: Label='{action_type.value}', Reward={reward_val}")
            
        final_score = task.grade(env.state())
        print(f"  [RESULT] Task Graded Score: {final_score:.2f}")

if __name__ == "__main__":
    run_minimal_test()
