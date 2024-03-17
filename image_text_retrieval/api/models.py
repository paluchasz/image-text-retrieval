import pydantic


class TextToImageRequestData(pydantic.BaseModel):
    text: str
