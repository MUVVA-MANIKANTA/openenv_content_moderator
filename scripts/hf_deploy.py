import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder

def deploy():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set.")
        sys.exit(1)

    api = HfApi(token=token)
    
    # Use your username to build the repo name
    user = api.whoami()["name"]
    repo_id = f"{user}/ai-social-guard"
    
    print(f"Deploying to Hugging Face: {repo_id}...")
    
    try:
        # 1. Create the Repo as a Space
        print("Creating Space...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True # Don't error if it already exists
        )
        
        # 2. Upload the entire project folder
        # Path to this project folder
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print(f"Uploading folder: {project_path} ...")
        upload_folder(
            folder_path=project_path,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            # Ignore binary files or caches
            ignore_patterns=["*.pyc", "__pycache__", ".git", ".venv", "venv", "*.rar", "*.csv"]
        )
        
        print(f"\nSUCCESS! Your AI Social Guard is deploying at:")
        print(f"https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    deploy()
