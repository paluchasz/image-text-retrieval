import functools
from pathlib import Path

import pydantic_settings

from image_text_retrieval import clip


class Settings(pydantic_settings.BaseSettings):
    image_text_retriever: clip.ImageTextRetriever


@functools.lru_cache()
def get_settings() -> Settings:
    image_text_retriever = clip.ImageTextRetriever(
        image_embeddings_path=Path("data/embeddings_all.npy"), index_to_image_mapping=Path("data/index_to_image_mapping_all.pkl")
    )  # Todo make env vars
    return Settings(image_text_retriever=image_text_retriever)
