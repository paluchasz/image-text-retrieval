import uvicorn


def main() -> None:
    uvicorn_settings = {"host": "0.0.0.0", "port": 8000}
    uvicorn.run("image_text_retrieval.api.app:create_api", **uvicorn_settings)


if __name__ == "__main__":
    main()
