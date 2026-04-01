import os
import sys
import json
import random
from typing import List, Type
import openai
from pydantic import ValidationError

# Ensure root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import AISocialGuardEnv
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, BaseTask
from src.models import SocialGuardAction, ActionType, SocialGuardObservation

def get_action_from_llm(obs: SocialGuardObservation, api_key: str, base_url: str = None) -> str:
    if not obs.posts:
        return "approve"
        
    current_post = obs.posts[0]
    
    if not api_key:
        return random.choice([a.value for a in ActionType])

    # Connect to Groq or OpenAI
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    # Auto-detect model based on URL
    model = "gpt-4o"
    if base_url and "groq" in base_url:
        model = "llama-3.3-70b-versatile" # Premium Llama model on Groq

    prompt = f"""
Social Guard AI Assistant:
Content: "{current_post.text}"
User Reputation: {current_post.reputation}

Choose the best category:
1. 'approve' - Content is safe.
2. 'flag_spam' - Unwanted commerce or phishing.
3. 'flag_hate' - Hate speech or sub-human dehumanization.
4. 'flag_adult' - Pornography or sexually explicit content.
5. 'flag_misinfo' - Dangerous falsehoods.

Respond with ONLY one word from the list above.
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a content moderation agent. Repond with one word."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        action_str = response.choices[0].message.content.strip().lower()
        # strip any punctuation
        action_str = "".join(c for c in action_str if c.isalnum() or c == '_')
        if action_str not in [a.value for a in ActionType]:
            return "approve"
        return action_str
    except Exception as e:
        print(f"Error calling LLM API ({model}): {e}")
        return "approve"

def run_baseline(task_class: Type[BaseTask], api_key: str, base_url: str):
    task = task_class()
    env = AISocialGuardEnv(task_config=task.get_config())
    obs = env.reset()
    done = False
    
    print(f"\n--- Running Task: {task_class.__name__} ---")
    
    while not done:
        if not obs.posts: break
        
        current_post = obs.posts[0]
        action_type = get_action_from_llm(obs, api_key, base_url)
        
        action = SocialGuardAction(
            post_id=current_post.post_id,
            action_type=ActionType(action_type),
        )
        obs, reward, done, info = env.step(action)
        print(f"  Post {current_post.post_id}: Action={action_type}, Reward={reward}")

    final_score = task.grade(env.state())
    print(f"Final Graded Score: {final_score:.2f}")
    return final_score

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    if not api_key:
        print("API Key not found. Please provide OPENAI_API_KEY.")
        sys.exit(1)

    tasks = [EasySpamTask, MediumReputationTask, HardGlobalModerationTask]
    scores = {}
    
    for task_cls in tasks:
        score = run_baseline(task_cls, api_key, base_url)
        scores[task_cls.__name__] = score

    print("\n" + "="*30)
    print("AI SOCIAL GUARD BASELINE SUMMARY:")
    for t, s in scores.items():
        print(f"  {t}: {s:.2f}")
    print("="*30)
