import pydantic


class TextToImageRequestData(pydantic.BaseModel):
    text: str


class Prediction(pydantic.BaseModel):
    text: str
    probability: float


class ImagetoTextResponse(pydantic.BaseModel):
    predictions: list[Prediction]
