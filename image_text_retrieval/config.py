from pathlib import Path

import pydantic_settings


class EnvVars(pydantic_settings.BaseSettings):
    IMAGE_EMBEDDINGS_PATH: Path = Path("data/embeddings/image_embeddings.npy")
    INDEX_TO_IMAGE_MAPPING: Path = Path("data/embeddings/index_to_image_mapping.pkl")
    TEXT_EMBEDDINGS_PATH: Path = Path("data/embeddings/text_embeddings.npy")
    INDEX_TO_TEXT_MAPPING: Path = Path("data/embeddings/index_to_text_mapping.pkl")


ENV_VARS = EnvVars()

__all__ = ["ENV_VARS"]
