import os
import sys
import random

# Ensure root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import AISocialGuardEnv
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask
from src.models import SocialGuardAction, ActionType

def run_minimal_test():
    # Use actual class instances from tasks.py
    tasks = [EasySpamTask(), MediumReputationTask(), HardGlobalModerationTask()]
    print("="*40)
    print("   AI SOCIAL GUARD: MINIMAL TEST RUN   ")
    print("="*40)
    
    for task in tasks:
        print(f"\n--- MISSION: {task.__class__.__name__} ---")
        env = AISocialGuardEnv(task_config=task.get_config())
        obs = env.reset()
        done = False
        
        while not done:
            if not obs.posts: break
            
            # Simulated Agent Choice
            current_post = obs.posts[0]
            # 70% chance to be a 'smart' agent, 30% random
            if random.random() < 0.7:
                action_type = current_post.correct_label
            else:
                action_type = random.choice([a.value for a in ActionType])
                
            action = SocialGuardAction(
                post_id=current_post.post_id,
                action_type=ActionType(action_type)
            )
            obs, reward, done, info = env.step(action)
            print(f"  [STEP] Post {current_post.post_id}: Label='{action_type}', Reward={reward}")
            
        final_score = task.grade(env.state())
        print(f"  [RESULT] Task Graded Score: {final_score:.2f}")

if __name__ == "__main__":
    run_minimal_test()
