import pickle
from pathlib import Path

import datasets
import numpy as np
from loguru import logger

from image_text_retrieval import clip


def main() -> None:
    logger.info("Running script")
    retriever = clip.ImageTextRetriever()

    image_dir = Path("data/images")
    image_paths = [path for path in image_dir.iterdir() if path.suffix == ".jpg"]
    image_paths = image_paths[:40]
    logger.info("Loading images")
    dataset = datasets.Dataset.from_dict({"image": [str(path) for path in image_paths]}).cast_column("image", datasets.Image())

    logger.info(f"Computing image embeddings for {len(image_paths)} images")
    embeddings = retriever.generate_image_embeddings(dataset["image"], batch_size=3)
    idx_to_image_mapping = dict(enumerate(image_paths))

    np.save("data/embeddings.npy", embeddings)
    with open("data/index_to_image_mapping.pkl", "wb") as file:
        pickle.dump(idx_to_image_mapping, file)


if __name__ == "__main__":
    main()
