import functools

import pydantic_settings

from image_text_retrieval import clip


class Settings(pydantic_settings.BaseSettings):  # type: ignore [valid-type, misc, unused-ignore]
    image_text_retriever: clip.ImageTextRetriever


@functools.lru_cache()
def get_settings() -> Settings:
    image_text_retriever = clip.ImageTextRetriever()
    return Settings(image_text_retriever=image_text_retriever)
