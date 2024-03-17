import fastapi

router = fastapi.APIRouter(tags=["health"], responses={404: {"description": "Not found"}})


@router.get("/ruok")
async def get_ruok() -> str:
    return "ok"
