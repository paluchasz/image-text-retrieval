import io
import random
from pathlib import Path

import datasets
import fastapi

from image_text_retrieval import settings
from image_text_retrieval.api import models

router = fastapi.APIRouter(tags=["health"], responses={404: {"description": "Not found"}})


@router.get("/ruok")
async def get_ruok() -> str:
    return "ok"


image_dir = Path("data/images")
image_paths = [path for path in image_dir.iterdir() if path.suffix == ".jpg"]

random.seed(0)
random.shuffle(image_paths)
image_paths = image_paths[:10]
dataset = datasets.Dataset.from_dict({"image": [str(path) for path in image_paths]}).cast_column("image", datasets.Image())


@router.post("/retrieve_image_from_text")
async def retrieve_image_from_text(request_data: models.TextToImageRequestData) -> fastapi.Response:
    predictions = settings.get_settings().image_text_retriever.predict_image_from_text([data["image"] for data in dataset], request_data.text)
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    img_byte_arr = io.BytesIO()
    predictions[0][0].save(img_byte_arr, format="PNG")
    return fastapi.Response(img_byte_arr.getvalue(), media_type="image/jpeg")
