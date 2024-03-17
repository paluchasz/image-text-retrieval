import fastapi
from loguru import logger

from image_text_retrieval.api import routes


def create_api() -> fastapi.FastAPI:
    logger.info("Creating Text-Image retrieval application")
    app = fastapi.FastAPI()
    app.include_router(routes.router)
    return app
