import io

import fastapi
import PIL.Image

from image_text_retrieval import settings
from image_text_retrieval.api import models

router = fastapi.APIRouter(tags=["health"], responses={404: {"description": "Not found"}})


@router.get("/ruok")
async def get_ruok() -> str:
    return "ok"


@router.post("/retrieve_image_from_text")
async def retrieve_image_from_text(request_data: models.TextToImageRequestData) -> fastapi.Response:
    predictions = settings.get_settings().image_text_retriever.predict_pre_computed_images_from_text(request_data.text)
    return fastapi.responses.FileResponse(predictions[0][0])


@router.post("/retrieve_text_from_image")
async def retrieve_text_from_image(file: fastapi.UploadFile, max_results: int = 10) -> models.ImagetoTextResponse:
    request_object_content = await file.read()
    image = PIL.Image.open(io.BytesIO(request_object_content))
    predictions = settings.get_settings().image_text_retriever.predict_pre_computed_text_from_image(image)
    return models.ImagetoTextResponse(
        predictions=[models.Prediction(text=text, probability=prob) for text, prob in predictions[: min(max_results, len(predictions))]]
    )
