"""Flask application factory."""

from flask import Flask

from src.utils import config


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["DEBUG"] = config.FLASK_DEBUG

    # Warm up the retrieval service at startup
    with app.app_context():
        from app.services import RetrievalService
        try:
            RetrievalService.get()
        except RuntimeError as e:
            import warnings
            warnings.warn(
                f"[startup] RetrievalService not ready: {e}\n"
                "Run `python setup_indexes.py` first.",
                stacklevel=1,
            )

    from app.routes import bp
    app.register_blueprint(bp)

    return app
