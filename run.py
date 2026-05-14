"""Flask application entry point."""

from dotenv import load_dotenv
load_dotenv()

from app import create_app
from src.utils import config

app = create_app()

if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
