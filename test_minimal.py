import random
from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType
from src.tasks import EasySpamTask

def test_run():
    env = AISocialGuardEnv()
    task = EasySpamTask()
    
    config = task.get_config()
    obs = env.reset(seed=42, task_config=config)
    rng = random.Random(42)
    
    gt = {p["post_id"]: ActionType(p["correct_label"]) for p in config["posts"]}
    
    done = False
    while not done:
        current_post = obs.current_post
        if not current_post:
            break
            
        post_id = current_post["post_id"]
        correct_label = gt.get(post_id, ActionType.APPROVE)
        
        if rng.random() < 0.7:
            action_type = correct_label
        else:
            action_type = rng.choice(list(ActionType))
            
        action = SocialGuardAction(
            post_id=post_id,
            action_type=action_type,
            reason="Randomized testing action"
        )
        
        obs, reward, done, info = env.step(action)
        reward_val = reward.value if hasattr(reward, 'value') else reward
        print(f"Step for post_id {post_id}: action={action.action_type.value}, reward={reward_val}")
        
    score = task.grade(env.state())
    print(f"Final Score: {score}")

if __name__ == "__main__":
    test_run()
