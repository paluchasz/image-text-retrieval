import pickle
from pathlib import Path

import datasets
import numpy as np
from loguru import logger

from image_text_retrieval import clip


def main() -> None:
    # Todo set up with DVC
    logger.info("Running script")
    retriever = clip.ImageTextRetriever()

    image_dir = Path("data/images")
    image_paths = [path for path in image_dir.iterdir() if path.suffix == ".jpg"]
    logger.info("Loading images")
    dataset = datasets.Dataset.from_dict({"image": [str(path) for path in image_paths]}).cast_column("image", datasets.Image())

    logger.info(f"Computing image embeddings for {len(image_paths)} images")
    image_embeddings = retriever.generate_image_embeddings(dataset["image"], batch_size=32)
    idx_to_image_mapping = dict(enumerate(image_paths))

    np.save("data/embeddings.npy", image_embeddings)
    with open("data/index_to_image_mapping.pkl", "wb") as file:
        pickle.dump(idx_to_image_mapping, file)


if __name__ == "__main__":
    main()
