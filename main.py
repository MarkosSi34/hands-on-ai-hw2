import os
import argparse

import uvicorn


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Start the Adult Income conversational-agent API."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "Google Gemini API key. Overrides GOOGLE_API_KEY from the shell / "
            ".env. Convenient, but visible in `ps` and shell history — prefer "
            ".env for anything sensitive."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1, or $HOST).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to listen on (default: 8000, or $PORT).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # If a key was passed on the CLI, put it in the environment BEFORE uvicorn
    # imports the app and the lifespan builds the agent. The agent reads
    # GOOGLE_API_KEY from the environment, and load_dotenv() does not override
    # an already-set var, so this wins over both the shell env and .env.
    if args.api_key:
        os.environ["GOOGLE_API_KEY"] = args.api_key

    # Import string (not the app object) so reload works if enabled.
    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=False)
