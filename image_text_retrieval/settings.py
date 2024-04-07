import functools

import pydantic_settings

from image_text_retrieval import config
from image_text_retrieval.ai import clip


class Settings(pydantic_settings.BaseSettings):
    image_text_retriever: clip.ImageTextRetriever


@functools.lru_cache()
def get_settings() -> Settings:
    image_text_retriever = clip.ImageTextRetriever(
        image_embeddings_path=config.ENV_VARS.IMAGE_EMBEDDINGS_PATH,
        index_to_image_mapping=config.ENV_VARS.INDEX_TO_IMAGE_MAPPING,
        text_embeddings_path=config.ENV_VARS.TEXT_EMBEDDINGS_PATH,
        index_to_text_mapping_path=config.ENV_VARS.INDEX_TO_TEXT_MAPPING,
    )
    return Settings(image_text_retriever=image_text_retriever)
