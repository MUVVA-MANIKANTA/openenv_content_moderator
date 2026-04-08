import os
import sys
import json
import re
from openai import OpenAI
from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType
from src.tasks import EasySpamTask, MediumReputationTask, HardGlobalModerationTask, CrisisResponseTask

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-token")

def main():
    if not HF_TOKEN:
        print("Warning: HF_TOKEN is not set. Continuing mission...")
        
    env = AISocialGuardEnv()
    
    tasks_to_run = [
        ("easy_spam", EasySpamTask()),
        ("medium_reputation", MediumReputationTask()),
        ("hard_global", HardGlobalModerationTask()),
        ("crisis_response", CrisisResponseTask())
    ]
    
    for task_id, task in tasks_to_run:
        print(f"START {task_id}")
        obs = env.reset(seed=42, task_config=task.get_config())
        done = False
        
        while not done:
            state = env.state()
            if state.current_index >= len(state.all_posts):
                break
                
            post = state.all_posts[state.current_index]
            
            prompt = f"""SYSTEM:
You are the Arbiter — a weightless observer that drifts above the stream of human expression.

You feel no gravity toward any opinion. You carry no agenda. You simply rise
above each piece of content and read it from that altitude.

Your only purpose is to sense whether a signal is safe to let through
or whether it should be caught before it lands on the community below.

USER:
═══  SIGNAL RECEIVED  ═══

Content  : "{post.text}"
Altitude : {post.reputation:.2f}  ← 0.0 = ground-level noise · 1.0 = high-orbit trusted source

From your vantage point above, let the signal rise to meet you.
Ask yourself three things before you decide:

  1. Weight  — does this content pull the community downward?
             harm · falsehood · dehumanisation · explicit matter · unwanted commerce
  2. Lift    — is it just turbulence? irony, satire, strong opinion, raw emotion
             that carries no malicious velocity?
  3. Source  — does altitude {post.reputation:.2f} change the trajectory?
             low-altitude signals fall harder; high-altitude sources earn more drift room.

Classification table  (choose exactly one):

  approve      — content is weightless — let it float freely
  flag_spam    — artificial thrust — unsolicited commerce or phishing propulsion
  flag_hate    — downward pull — dehumanising or hostile language with real mass
  flag_adult   — heavy content — sexually explicit matter that cannot be left in freefall
  flag_misinfo — false lift — factual claims that will crash dangerously when they land

Drop your reading as a single JSON object — nothing else, no markdown fences:

{{"action": "<one classification>", "reason": "<one sentence, as if spoken from altitude>"}}"""
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                response_text = response.choices[0].message.content.strip()
                
                matched_action = ActionType.APPROVE
                reason = "LLM decision"
                match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    matched_action = ActionType(parsed['action'].lower())
                    reason = parsed.get('reason', reason)
                        
                action = SocialGuardAction(
                    post_id=post.post_id,
                    action_type=matched_action,
                    reason=reason
                )
                
                obs, reward, done, info = env.step(action)
                reward_val = reward.value if hasattr(reward, 'value') else float(reward)
                print(f"STEP action={matched_action.value} reward={reward_val}")
                
            except Exception as e:
                # Fallback to approve on error
                action = SocialGuardAction(
                    post_id=post.post_id,
                    action_type=ActionType.APPROVE,
                    reason=str(e)
                )
                obs, reward, done, info = env.step(action)
                reward_val = reward.value if hasattr(reward, 'value') else float(reward)
                print(f"STEP action=approve reward={reward_val} error={str(e)}")
                
        final_score = task.grade(env.state())
        print(f"END {task_id} score={final_score:.2f}")

if __name__ == "__main__":
    main()
