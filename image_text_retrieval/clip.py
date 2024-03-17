import PIL.Image
import torch
import transformers
from loguru import logger


def predict_image_from_text(
    images: list[PIL.Image.Image], description: str, model: transformers.CLIPModel, processor: transformers.CLIPProcessor
) -> list[tuple[PIL.Image.Image, float]]:
    logger.info(f"Searching for images with description: {description}")
    with torch.no_grad():
        inputs = processor(text=description, images=images, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text
    probs = logits_per_text.softmax(dim=1)
    return list(zip(images, probs[0].tolist()))
