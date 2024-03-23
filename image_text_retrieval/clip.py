import more_itertools
import numpy as np
import numpy.typing as npt
import PIL.Image
import torch
import tqdm
import transformers
from loguru import logger


class ImageTextRetriever:
    def __init__(self) -> None:
        logger.info("Loading CLIP model and processor")
        self.model: transformers.CLIPModel = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor: transformers.CLIPProcessor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict_image_from_text(self, images: list[PIL.Image.Image], description: str) -> list[tuple[PIL.Image.Image, float]]:
        logger.info(f"Searching for images with description: {description}")
        with torch.no_grad():
            inputs = self.processor(text=description, images=images, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
        logits_per_text = outputs.logits_per_text
        probs = logits_per_text.softmax(dim=1)
        return list(zip(images, probs[0].tolist()))

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
