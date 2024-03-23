import PIL.Image
import torch
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
