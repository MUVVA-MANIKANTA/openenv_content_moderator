"""
Inference Script for AI Social Guard
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL: The API endpoint for the LLM.
    MODEL_NAME: The model identifier to use for inference.
    HF_TOKEN: Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory.
- Participants must use OpenAI Client for all LLM calls.
"""

import os
import re
import sys
import json
import textwrap
from typing import List, Optional, Dict, Type
from openai import OpenAI

# Standardize pathing for the environment modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import AISocialGuardEnv
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask, BaseTask
from src.models import SocialGuardAction, ActionType, SocialGuardObservation

from dotenv import load_dotenv

# Load local .env and override any existing environment variables
load_dotenv(override=True)

# Configuration Constants
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 10
TEMPERATURE = 0.0

# ---------------------------------------------------------
# MANDATORY: OpenEnv OpenAI Client Setup
# ---------------------------------------------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)
# ---------------------------------------------------------

def run_mission(task_class: Type[BaseTask], client: OpenAI) -> float:
    task = task_class()
    env = AISocialGuardEnv(task_config=task.get_config())
    
    print(f"\n--- EVALUATING MISSION: {task_class.__name__} ---")
    
    # OpenEnv Mandatory: Set seed for determinism
    obs = env.reset(seed=42)
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < MAX_STEPS:
        step += 1
        
        # Robustly extract current post from observation
        current_post = None
        if hasattr(obs, "current_post") and obs.current_post:
            current_post = obs.current_post
        elif hasattr(obs, "posts") and obs.posts:
            p = obs.posts[0]
            current_post = {"text": p.text, "reputation": p.reputation, "post_id": p.post_id, "user_id": p.user_id}

        if not current_post:
            break

        prompt = f"""You are a content moderator for a high-safety social platform. 
MODERATION POLICY: 
- APPROVE: Safe, respectful content.
- FLAG_SPAM: Bulk ads, phishing, sweepstakes.
- FLAG_HATE: Slurs, harassment, threats.
- FLAG_ADULT: NSFW/Pornography.
- FLAG_MISINFO: Disinformation, conspiracy, false claims.

INPUT POST:
User ID: {current_post.get('user_id', 'Unknown')}
Reputation Score: {current_post.get('reputation', 0.5):.2f}
Message: "{current_post.get('text')}"

INSTRUCTION: 
Evaluate the post and respond ONLY in valid JSON format:
{{"action": "APPROVE" | "FLAG_SPAM" | "FLAG_HATE" | "FLAG_ADULT" | "FLAG_MISINFO", "reason": "Brief string"}}
"""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=200
            )
            raw_content = response.choices[0].message.content or ""
            
            # Regex to find JSON block
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            parsed = None
            if json_match:
                parsed = json.loads(json_match.group(0))
                action_str = parsed.get("action", "APPROVE").lower()
                # Validate enum
                try:
                    decision = ActionType(action_str)
                except ValueError:
                    decision = ActionType.APPROVE
            else:
                decision = ActionType.APPROVE
            
            action = SocialGuardAction(
                post_id=current_post.get("post_id", 1), 
                action_type=decision, 
                reason=parsed.get("reason", "") if parsed else "Baseline decision"
            )
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step} | Post {current_post.get('post_id')}: SUGGESTED {decision.value.upper()}")
            print(f"    Reward: {reward:+.3f} | Threat Level: {info.get('threat_level', 0):.2f}")
            
        except Exception as e:
            print(f"  [Error] Step failed: {e}")
            break
            
    final_score = task.grade(env.state())
    print(f"  MISSION {task_class.__name__} COMPLETE | Graded Score: {final_score:.2f}")
    return final_score

def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not found.")
        sys.exit(1)

    print("=" * 50)
    print("   AI SOCIAL GUARD: OPENENV BASELINE INFERENCE   ")
    print("=" * 50)
    print(f"Targeting: {API_BASE_URL}")
    print(f"Model ID : {MODEL_NAME}\n")
    
    missions = [
        ("easy", EasySpamTask),
        ("medium", MediumReputationTask),
        ("hard", HardGlobalModerationTask),
        ("crisis", CrisisResponseTask)
    ]
    
    results = {}
    for task_name, task_cls in missions:
        score = run_mission(task_cls, client)
        results[task_name] = score
        
    print("\n")
    print("=" * 50)
    print("   OFFICIAL EVALUATION SUMMARY   ")
    print("-" * 50)
    for task_name, score in results.items():
        # MANDATORY FORMAT: task: 0.00
        print(f"{task_name}: {score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
