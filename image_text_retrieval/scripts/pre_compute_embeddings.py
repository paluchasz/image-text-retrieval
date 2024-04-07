import pickle

import datasets
import dvc.api
import numpy as np
import pandas as pd
from loguru import logger

from image_text_retrieval import params_models
from image_text_retrieval.ai import clip


def main() -> None:
    logger.info("Running script to pre-compute embeddings into a file")
    dvc_params = dvc.api.params_show()
    params = params_models.PreComputeEmbeddingsParams(**dvc_params)

    retriever = clip.ImageTextRetriever()
    logger.info("Loaded retriever")

    image_paths = [path for path in params.image_dir.iterdir() if path.suffix in {".jpg", ".jpeg", ".png"}]
    logger.info("Loading images")
    dataset = datasets.Dataset.from_dict({"image": [str(path) for path in image_paths]}).cast_column("image", datasets.Image())
    logger.info("Loading captions")
    captions_df = pd.read_csv(params.captions_file_path)

    logger.info(f"Computing image embeddings for {len(image_paths)} images")
    image_embeddings = retriever.generate_image_embeddings(dataset["image"], batch_size=params.batch_size)
    index_to_image_mapping = dict(enumerate(image_paths))

    logger.info("Computing embeddings for captions")
    text_embeddings = retriever.generate_text_embeddings(captions_df.caption.to_list(), batch_size=params.batch_size)
    index_to_caption_mapping = dict(enumerate(captions_df.caption))

    logger.info(f"Saving out embeddings and mappings in {params.output_dir}")
    params.output_dir.mkdir(exist_ok=True, parents=True)
    np.save(params.image_embeddings_file_path, image_embeddings)
    with open(params.index_to_mapping_file_path, "wb") as file:
        pickle.dump(index_to_image_mapping, file)

    np.save(params.text_embeddings_file_path, text_embeddings)
    with open(params.index_to_text_mapping_file_path, "wb") as file:
        pickle.dump(index_to_caption_mapping, file)


if __name__ == "__main__":
    main()
