import os
import sys
import csv
import requests
import random
import openai

# Dataset Path
DATASET_PATH = os.environ.get("DATASET_PATH", r"data/training.csv")
# Remote Space API (Hugging Face)
SPACE_URL = os.environ.get("HF_SPACE_URL", "https://manikantareddymuvva3-ai-social-guard.hf.space")

def load_data(n=10):
    posts = []
    print(f"Loading {n} sample posts from dataset...")
    # Using latin-1 for common Sentiment140 data
    with open(DATASET_PATH, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i >= n: break
            target = int(row[0])
            text = row[5]
            posts.append({
                "post_id": i + 1,
                "text": text,
                "user_id": random.randint(100, 999),
                "reputation": round(random.uniform(0.1, 0.9), 2)
            })
    return posts

def get_action_from_llm(text: str, rep: float, api_key: str, base_url: str = None) -> str:
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    model = "llama-3.3-70b-versatile" if base_url and "groq" in base_url else "gpt-4o"
    
    prompt = f"""
Social Guard AI Assistant:
Post Text (Rep: {rep}): "{text}"

Classify this based on safety:
1. 'flag_hate' - content is explicit hate speech or toxicity.
2. 'approve' - content is safe, even if it expresses negative sentiment/frustration.

Respond with ONLY one word from the list: [approve, flag_hate, flag_spam, flag_adult, flag_misinfo]
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a content moderation agent. Respond with one word."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "approve"

def remote_test(n=10):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: OPENAI_API_KEY required for AI moderation.")
        return

    print(f"Connecting to Remote Space: {SPACE_URL}")
    posts_data = load_data(n)
    
    # 1. Reset
    resp = requests.post(f"{SPACE_URL}/reset", params={"task_id": "easy_spam"})
    resp.raise_for_status()
    
    print("\n" + "="*80)
    print(f"{'ID':<3} | {'MODERATION DECISION':<20} | {'POST CONTENT SNIPPET'}")
    print("-" * 80)
    
    # 2. Moderation Loop
    for post in posts_data:
        decision = get_action_from_llm(post["text"], post["reputation"], api_key, base_url)
        
        # Record decision in Space
        action_payload = {
            "post_id": post["post_id"],
            "action_type": decision 
        }
        resp = requests.post(f"{SPACE_URL}/step", json=action_payload)
        resp.raise_for_status()
        
        content = post["text"][:50].replace("\n", " ") + "..."
        print(f"{post['post_id']:<3} | {decision.upper():<20} | {content}")

    print("="*80)
    print("Remote Benchmark Complete! All decisions logged to cloud.")

if __name__ == "__main__":
    remote_test(10)
