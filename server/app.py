import uvicorn
import os
import sys

# Add the parent directory to sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import app

def main():
    """
    Main entry point for the OpenEnv multi-mode deployment.
    This function is mapped to the 'server' command in pyproject.toml.
    """
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting AI Social Guard server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
