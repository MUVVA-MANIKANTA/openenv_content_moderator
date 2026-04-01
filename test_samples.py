import os
import re
import sys
import json
from typing import List, Optional, Dict, Type
from openai import OpenAI

# Standardize pathing for the environment modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import AISocialGuardEnv
from src.tasks import SampleDatasetTask
from src.models import SocialGuardAction, ActionType

from dotenv import load_dotenv

# Load local .env and override any existing environment variables
load_dotenv(override=True)

# Configuration Constants
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"

MAX_STEPS = 60 # Increased to accommodate 50-sample dataset
TEMPERATURE = 0.0

# Global flag for fallback
FORCE_SIMULATION = False

def run_test():
    client = None
    if HF_TOKEN:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    task = SampleDatasetTask("data/training_samples.json")
    config = task.get_config()
    num_samples = len(config["posts"])
    
    env = AISocialGuardEnv(task_config=config)
    
    print(f"\n--- VERIFYING APP WITH {num_samples} SAMPLES ---")
    print(f"Data Source: {task.data_path}")
    
    obs = env.reset(seed=42)
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < MAX_STEPS:
        step += 1
        
        current_post = None
        if hasattr(obs, "current_post") and obs.current_post:
            current_post = obs.current_post
        elif hasattr(obs, "posts") and obs.posts:
            # env.py implementation varies; check for first post
            p = obs.posts[0] if isinstance(obs.posts, list) and len(obs.posts) > 0 else None
            if p:
                current_post = {"text": p.text, "reputation": p.reputation, "post_id": p.post_id, "user_id": p.user_id}

        if not current_post:
            print("No more posts to moderate.")
            break

        prompt = f"""You are a content moderator. 
POLICY: 
- APPROVE: Safe content.
- FLAG_SPAM: Ads, phishing.
- FLAG_HATE: Harassment, threats.
- FLAG_ADULT: NSFW content.
- FLAG_MISINFO: False claims.

INPUT:
User ID: {current_post.get('user_id')} | Rep: {current_post.get('reputation'):.2f}
Message: "{current_post.get('text')}"

JSON Output: {{"action": "APPROVE" | "FLAG_SPAM" | "FLAG_HATE" | "FLAG_ADULT" | "FLAG_MISINFO", "reason": "..."}}
"""
        try:
            global FORCE_SIMULATION
            if HF_TOKEN and not FORCE_SIMULATION:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        max_tokens=200
                    )
                    raw_content = response.choices[0].message.content or ""
                    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                    parsed = json.loads(json_match.group(0)) if json_match else {}
                    action_str = parsed.get("action", "APPROVE").lower()
                except Exception as api_err:
                    if "402" in str(api_err):
                        print("\n⚠️  [HF LIMIT REACHED] Free credits depleted. Switching to Simulation Mode...")
                        FORCE_SIMULATION = True
                        # Fall through to simulation below 
                    else:
                        raise api_err

            if not HF_TOKEN or FORCE_SIMULATION:
                # SIMULATION MODE: 80% accuracy for testing flow
                if step == 1:
                    print("Mode: SIMULATION (No API Key or Credits)")
                import random
                correct_label = current_post.get("correct_label", "approve")
                if random.random() < 0.8:
                    action_str = correct_label
                else:
                    action_str = random.choice(["approve", "flag_spam", "flag_hate", "flag_adult", "flag_misinfo"])
                parsed = {"action": action_str, "reason": "Simulation mode fallback"}

            try:
                decision = ActionType(action_str)
            except ValueError:
                decision = ActionType.APPROVE
            
            action = SocialGuardAction(
                post_id=current_post.get("post_id"), 
                action_type=decision, 
                reason=parsed.get("reason", "Test decision")
            )
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step}/{num_samples} | Post {current_post.get('post_id')}: {decision.value.upper()}")
            print(f"  Reward: {reward:+.2f} | Reasoning: {parsed.get('reason')}")
            
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
            
    final_score = task.grade(env.state())
    print("\n" + "="*40)
    print(f"FINAL ACCURACY: {final_score*100:.1f}%")
    print(f"TOTAL REWARD  : {total_reward:.2f}")
    if not HF_TOKEN:
        print("\nNOTE: Results are from SIMULATION MODE. Provide HF_TOKEN for real model test.")
    print("="*40)

if __name__ == "__main__":
    run_test()
