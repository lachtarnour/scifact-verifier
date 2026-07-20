"""Flask application entry point."""

from app import create_app
from src.config import config

app = create_app()

if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
