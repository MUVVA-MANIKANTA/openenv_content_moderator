import os
import sys
import json
from openai import OpenAI
from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token")

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN or OPENAI_API_KEY is not set")
        sys.exit(1)
        
    env = AISocialGuardEnv()
    
    tasks_to_run = [
        ("easy_spam", EasySpamTask()),
        ("medium_reputation", MediumReputationTask()),
        ("hard_global", HardGlobalModerationTask()),
        ("crisis_response", CrisisResponseTask())
    ]
    
    for task_id, task in tasks_to_run:
        obs = env.reset(seed=42, task_config=task.get_config())
        done = False
        
        while not done:
            state = env.state()
            if state.current_index >= len(state.all_posts):
                break
                
            post = state.all_posts[state.current_index]
            
            prompt = f"Analyze this post and categorize it. Text: '{post.text}'. User reputation: {post.reputation}. Respond with exactly one of: approve, flag_spam, flag_hate, flag_adult, flag_misinfo."
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip().lower()
                
                matched_action = ActionType.APPROVE
                for a in ActionType:
                    if a.value in response_text:
                        matched_action = a
                        break
                        
                action = SocialGuardAction(
                    post_id=post.post_id,
                    action_type=matched_action,
                    reason="Inferred from LLM"
                )
                
                obs, reward, done, info = env.step(action)
                reward_val = reward.value if hasattr(reward, 'value') else float(reward)
                
            except Exception as e:
                # Fallback to approve on error
                action = SocialGuardAction(
                    post_id=post.post_id,
                    action_type=ActionType.APPROVE,
                    reason=str(e)
                )
                obs, reward, done, info = env.step(action)
                
        final_score = task.grade(env.state())
        print(f"{task_id}: {final_score:.2f}")

if __name__ == "__main__":
    main()
