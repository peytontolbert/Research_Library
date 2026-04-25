"""
Convenience entrypoint to run the Repository Library FastAPI server.

Usage (from the project root):

    python server.py

This will start uvicorn with sensible defaults without requiring any
command-line arguments.
"""

from run import app
import uvicorn


def main() -> None:
    """
    Start the uvicorn server for the Repository Library API.
    """
    uvicorn.run(app, host="0.0.0.0", port=8011)


if __name__ == "__main__":
    main()


