import os
import sys
import csv
import random
import openai

# Path to the dataset
DATASET_PATH = os.environ.get("DATASET_PATH", r"data/training.csv")

# Add parent directory of scr to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import AISocialGuardEnv
from src.models import SocialGuardAction, ActionType, SocialGuardObservation, SocialGuardState

def load_data(n=20, offset=0):
    posts = []
    # encoding='latin-1' is common for this dataset
    print(f"Loading {n} posts (offset={offset}) from {DATASET_PATH}...")
    with open(DATASET_PATH, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        # Skip offset rows
        for _ in range(offset):
            next(reader, None)
            
        for i, row in enumerate(reader):
            if i >= n: break
            # Sentiment140 mapping: 0:negative, 4:positive.
            # IN CONTENT MODERATION, most negative sentiment is SAFE to approve.
            # We map most to APPROVE, unless they are truly toxic (but we lack that label).
            # For the benchmark simulation, we'll mark some negative as HATE for testing.
            target = int(row[0])
            text = row[5]
            
            # Simple simulation: Every 5th negative post is "hate" for benchmark GT
            if target == 0 and (i % 5 == 0):
                correct_label = ActionType.FLAG_HATE
            else:
                correct_label = ActionType.APPROVE
                
            posts.append({
                "post_id": i + 1 + offset,
                "text": text,
                "user_id": random.randint(100, 999),
                "reputation": round(random.uniform(0.1, 0.9), 2),
                "correct_label": correct_label
            })
    return posts

def get_action_from_llm(obs: SocialGuardObservation, api_key: str, base_url: str = None) -> str:
    if not obs.posts: return "approve"
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    current_post = obs.posts[0]
    model = "llama-3.3-70b-versatile" if base_url and "groq" in base_url else "gpt-4o"
    
    prompt = f"""
### AI Social Guard Moderator Interface
Current Post Context:
- User Reputation Score: {current_post.reputation} (Low = Suspicious, High = Trusted)
- Post Content: "{current_post.text}"

Policy Definitions:
- flag_hate: Use ONLY for slurs, harassment, or explicit dehumanizing speech. Sad/Angry sentiment is NOT hate.
- flag_spam: Promotions, giveaways, or phishing.
- flag_adult: NSFW content.
- flag_misinfo: Factually dangerous claims.
- approve: Safe for the community (includes normal opinions, sadness, or mild disagreement).

Determine the appropriate action. Respond with ONLY one word: [approve, flag_hate, flag_spam, flag_adult, flag_misinfo]
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a professional content moderator with high sensitivity to false positives."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        action_str = response.choices[0].message.content.strip().lower()
        if action_str not in [a.value for a in ActionType]:
            return "approve"
        return action_str
    except Exception as e:
        print(f"Error calling LLM for Post {current_post.post_id}: {e}")
        return "approve"

def run_benchmark(n=100):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print("API Key required.")
        return

    # Mix 50 from start (negative) and 50 from near end (positive)
    print("Preparing 100 rows for moderation bench...")
    posts_data = load_data(50, offset=0) # Negative sentiment
    posts_data += load_data(50, offset=1500000) # Positive sentiment (near end)
    
    env = AISocialGuardEnv(task_config={"posts": posts_data})
    obs = env.reset()
    done = False
    
    print("\n--- Starting 100-Post Moderation Benchmark ---")
    results = []
    correct = 0
    total = len(posts_data)
    
    while not done:
        curr_post = obs.posts[0]
        action_type = get_action_from_llm(obs, api_key, base_url)
        action = SocialGuardAction(
            post_id=curr_post.post_id,
            action_type=ActionType(action_type)
        )
        obs, reward, done, info = env.step(action)
        
        # Ground Truth check
        # We find previous data for GT
        gt = next(p["correct_label"] for p in posts_data if p["post_id"] == curr_post.post_id)
        if action_type == gt: correct += 1
        
        results.append(f"Post {curr_post.post_id}: [{action_type.upper()}] Reward: {reward}")
        # print progress every 10
        if len(results) % 10 == 0:
            print(f"  Processed {len(results)}/100...")

    final_acc = (correct / total) * 100
    print(f"\nBenchmark Summary ({total} samples):")
    print(f"  Final Accuracy (against sentiment/safety mapping): {final_acc:.2f}%")
    print("\nEnvironment state:")
    moderated_count = len(env.state().actions_taken)
    print(f"  Total posts: {moderated_count}")

if __name__ == "__main__":
    run_benchmark(100)
