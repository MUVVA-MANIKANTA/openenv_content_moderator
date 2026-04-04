"""
Inference Script for AI Social Guard (OpenEnv Compliant)
======================================================
This script interacts with the AI Social Guard Environment directly.
It uses an LLM to moderate posts and evaluates performance across Easy, Medium, Hard, and Crisis tasks.
"""

import os
import re
import sys
import json
import requests
from typing import List, Optional, Dict, Any, Type
from openai import OpenAI
from dotenv import load_dotenv

from src.env import AISocialGuardEnv
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask
from src.models import SocialGuardAction, ActionType

# Load environment variables
load_dotenv(override=True)

# LLM Configuration
LLM_API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

MAX_STEPS = 20
TEMPERATURE = 0.0

# Initialize OpenAI Client for LLM
client = OpenAI(
    base_url=LLM_API_BASE_URL,
    api_key=HF_TOKEN or "no-token"
)

TASKS = {
    "easy_spam": EasySpamTask,
    "medium_reputation": MediumReputationTask,
    "hard_global": HardGlobalModerationTask,
    "crisis_response": CrisisResponseTask
}

def run_task(task_id: str) -> float:
    print(f"\n--- EVALUATING TASK: {task_id.upper()} ---")
    
    task = TASKS[task_id]()
    env = AISocialGuardEnv(task_config=task.get_config())
    obs = env.reset(seed=42)
    done = obs.done

    step = 0
    total_reward = 0
    
    while not done and step < MAX_STEPS:
        step += 1
        
        current_post = obs.current_post
        if not current_post:
            print("  [Info] No more posts in observation.")
            break

        # Construct Prompt for LLM
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
Your task is to provide two parts in your response:
1. "action": One of the approved labels.
2. "reason": A detailed explanation (at least 25 characters) of *why* this policy applies, citing specific content in the post.

Respond ONLY in valid JSON format:
{{
  "action": "APPROVE" | "FLAG_SPAM" | "FLAG_HATE" | "FLAG_ADULT" | "FLAG_MISINFO",
  "reason": "Detailed string explaining the violation or lack thereof..."
}}
"""
        try:
            # LLM Call
            llm_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=200
            )
            raw_content = llm_response.choices[0].message.content or ""
            print(f"  [Debug] LLM Response: {raw_content}")
            
            # Extract JSON from LLM response
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                action_str = parsed.get("action", "APPROVE").lower()
                reason = parsed.get("reason", "Moderated by AI")
                print(f"  [Debug] Parsed Action: {action_str}")
                try:
                    action_type = ActionType(action_str)
                except ValueError:
                    action_type = ActionType.APPROVE
            else:
                action_type = ActionType.APPROVE
                reason = "Default approval (JSON parsing failed)"
                print("  [Debug] JSON match failed.")
            
            action = SocialGuardAction(
                post_id=current_post.get("post_id"),
                action_type=action_type,
                reason=reason
            )
            
            obs, reward, done, info = env.step(action)
            
            reward_val = reward.value if hasattr(reward, 'value') else reward
            total_reward += float(reward_val)
            print(f"  Step {step} | Post {current_post.get('post_id')}: {action_type.value.upper()} | Reward: {reward_val:+.2f}")
            
        except Exception as e:
            print(f"  [Error] Step {step} failed: {e}")
            break
            
    final_score = task.grade(env.state())
    print(f"  TASK {task_id.upper()} COMPLETE | Accumulated Score: {final_score:.2f}")
    return final_score

def main():
    print("=" * 60)
    print("   AI SOCIAL GUARD: OPENENV API-BASED INFERENCE   ")
    print("=" * 60)
    print(f"LLM API  : {LLM_API_BASE_URL}")
    print(f"Model ID : {MODEL_NAME}")
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not found.")
        sys.exit(1)

    tasks_to_run = ["easy_spam", "medium_reputation", "hard_global", "crisis_response"]
    results = {}
    
    for t_id in tasks_to_run:
        score = run_task(t_id)
        results[t_id] = score
        
    print("\n" + "=" * 60)
    print("   OFFICIAL EVALUATION SUMMARY   ")
    print("-" * 60)
    for t_id, score in results.items():
        print(f"{t_id}: {score:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
