from pathlib import Path

import pydantic_settings


class EnvVars(pydantic_settings.BaseSettings):
    IMAGE_EMBEDDINGS_PATH: Path = Path("data/embeddings_all.npy")
    INDEX_TO_IMAGE_MAPPING: Path = Path("data/index_to_image_mapping_all.pkl")


ENV_VARS = EnvVars()

__all__ = ["ENV_VARS"]
