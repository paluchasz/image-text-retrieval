import typing as t
from pathlib import Path

import pydantic


class PreComputeEmbeddingsParams(pydantic.BaseModel):
    image_dir: Path
    captions_file_path: Path
    output_dir: Path
    image_embeddings_file_name: str
    index_to_image_mapping_file_name: str
    text_embeddings_file_name: str
    index_to_text_mapping_file_name: str
    batch_size: pydantic.PositiveInt

    @pydantic.model_validator(mode="before")
    @classmethod
    def _decompose(cls, values: dict[str, t.Any]) -> dict[str, t.Any]:
        transformed: dict[str, t.Any] = values["pre_compute_embeddings"]
        return transformed

    @property
    def image_embeddings_file_path(self) -> Path:
        return Path(self.output_dir / self.image_embeddings_file_name)

    @property
    def index_to_mapping_file_path(self) -> Path:
        return Path(self.output_dir / self.index_to_image_mapping_file_name)

    @property
    def text_embeddings_file_path(self) -> Path:
        return Path(self.output_dir / self.text_embeddings_file_name)

    @property
    def index_to_text_mapping_file_path(self) -> Path:
        return Path(self.output_dir / self.index_to_text_mapping_file_name)
