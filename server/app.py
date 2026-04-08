from app.main import app
import uvicorn

def main():
    """Main entry point for OpenEnv validation."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
