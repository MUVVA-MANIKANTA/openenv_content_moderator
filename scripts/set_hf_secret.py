import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

def set_secret():
    # Correct path to the .env in the parent root folder
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=True)
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in .env file.")
        return

    api = HfApi(token=token)
    repo_id = "reddymuvva3/ai-social-guard"
    
    print(f"Setting HF_TOKEN secret for {repo_id}...")
    try:
        api.add_space_secret(
            repo_id=repo_id,
            key="HF_TOKEN",
            value=token
        )
        print("✅ SUCCESS! The secret has been added to your Hugging Face Space.")
        print("Your Space will now restart automatically to apply this change.")
    except Exception as e:
        print(f"❌ Failed to set secret: {e}")

if __name__ == "__main__":
    set_secret()
