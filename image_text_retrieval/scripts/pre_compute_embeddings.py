import pickle

import datasets
import dvc.api
import numpy as np
from loguru import logger

from image_text_retrieval import clip, params_models


def main() -> None:
    logger.info("Running script to pre-compute embeddings into a file")
    dvc_params = dvc.api.params_show()
    params = params_models.PreComputeEmbeddingsParams(**dvc_params)

    retriever = clip.ImageTextRetriever()
    logger.info("Loaded retriever")

    image_paths = [path for path in params.image_dir.iterdir() if path.suffix in {".jpg", ".jpeg", ".png"}]
    logger.info("Loading images")
    dataset = datasets.Dataset.from_dict({"image": [str(path) for path in image_paths]}).cast_column("image", datasets.Image())

    logger.info(f"Computing image embeddings for {len(image_paths)} images")
    image_embeddings = retriever.generate_image_embeddings(dataset["image"], batch_size=32)
    idx_to_image_mapping = dict(enumerate(image_paths))

    logger.info(f"Saving out embeddings and mappings in {params.output_dir}")
    params.output_dir.mkdir(exist_ok=True, parents=True)
    np.save(params.image_embeddings_file_path, image_embeddings)
    with open(params.index_to_mapping_file_path, "wb") as file:
        pickle.dump(idx_to_image_mapping, file)


if __name__ == "__main__":
    main()
