import pickle
from pathlib import Path

import more_itertools
import numpy as np
import numpy.typing as npt
import PIL.Image
import scipy.special as scipy_special
import torch
import tqdm
import transformers
from loguru import logger


class ImageTextRetriever:
    def __init__(
        self,
        image_embeddings_path: Path | None = None,
        index_to_image_mapping: Path | None = None,
        text_embeddings_path: Path | None = None,
        index_to_text_mapping_path: Path | None = None,
    ) -> None:
        logger.info("Loading CLIP model and processor")
        self.model: transformers.CLIPModel = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor: transformers.CLIPProcessor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if image_embeddings_path and index_to_image_mapping:
            logger.info("Loading pre-computed images embeddings")
            self.image_embeddings = np.load(image_embeddings_path)
            self.image_embeddings /= np.linalg.norm(self.image_embeddings, ord=2, axis=-1)[:, None]
            with open(index_to_image_mapping, "rb") as file:
                self.index_to_image_mapping = pickle.load(file)  # noqa: S301
        if text_embeddings_path and index_to_text_mapping_path:
            logger.info("Loading pre-computed text embeddings")
            self.text_embeddings = np.load(text_embeddings_path)
            self.text_embeddings /= np.linalg.norm(self.text_embeddings, ord=2, axis=-1)[:, None]
            with open(index_to_text_mapping_path, "rb") as file:
                self.index_to_text_mapping = pickle.load(file)  # noqa: S301

    def predict_pre_computed_images_from_text(self, description: str) -> list[tuple[Path, float]]:
        logger.info(f"Searching for images with description: {description}")
        text_embeddings = self.generate_text_embeddings(descriptions=[description])
        text_embeddings /= np.linalg.norm(text_embeddings, ord=2, axis=-1)[:, None]
        distances = np.dot(self.image_embeddings, text_embeddings.T)
        distances *= np.exp(self.model.logit_scale.item())
        probabilities = scipy_special.softmax(distances)
        predictions = [(image_path, float(probs[0])) for image_path, probs in zip(self.index_to_image_mapping.values(), probabilities.tolist())]
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def predict_image_from_text(self, images: list[PIL.Image.Image], description: str) -> list[tuple[PIL.Image.Image, float]]:
        logger.info(f"Searching for images with description: {description}")
        with torch.no_grad():
            inputs = self.processor(text=description, images=images, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
        logits_per_text = outputs.logits_per_text
        probs = logits_per_text.softmax(dim=1)
        return sorted(zip(images, probs[0].tolist()), key=lambda x: x[1], reverse=True)

    def predict_pre_computed_text_from_image(self, image: PIL.Image.Image) -> list[tuple[str, float]]:
        logger.info("Searching for text with the image")
        image_embeddings = self.generate_image_embeddings(images=[image])
        image_embeddings /= np.linalg.norm(image_embeddings, ord=2, axis=-1)[:, None]
        distances = np.dot(image_embeddings, self.text_embeddings.T)
        distances *= np.exp(self.model.logit_scale.item())
        probabilities = scipy_special.softmax(distances)
        predictions = list(zip(self.index_to_text_mapping.values(), probabilities[0]))
        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def predict_text_from_image(self, image: PIL.Image.Image, descriptions: list[str]) -> list[tuple[str, float]]:
        with torch.no_grad():
            inputs = self.processor(text=descriptions, images=image, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # the logits are simply the dot products between the image and text embeddings
        probs = logits_per_image.softmax(dim=1)  # take softmax to get probabilities for each description
        return list(zip(descriptions, probs[0].tolist()))

    def generate_text_embeddings(self, descriptions: list[str], batch_size: int = 32) -> npt.NDArray[np.float64]:
        embeddings = []
        with torch.no_grad():
            for descriptions_batch in tqdm.tqdm(more_itertools.chunked(descriptions, batch_size), total=len(descriptions) / batch_size):
                inputs = self.processor(text=descriptions_batch, return_tensors="pt", padding=True, truncation=True)
                embeddings.append(self.model.get_text_features(**inputs).numpy())

        return np.concatenate(embeddings)

    def generate_image_embeddings(self, images: list[PIL.Image.Image], batch_size: int = 32) -> npt.NDArray[np.float64]:
        embeddings = []
        with torch.no_grad():
            for images_batch in tqdm.tqdm(more_itertools.chunked(images, batch_size), total=len(images) / batch_size):
                inputs = self.processor(images=images_batch, return_tensors="pt", padding=True, truncation=True)
                embeddings.append(self.model.get_image_features(**inputs).numpy())

        return np.concatenate(embeddings)
