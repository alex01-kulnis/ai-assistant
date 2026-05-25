from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.chat_langchain import router as chat_langchain_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.llm import router as llm_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(llm_router)
    app.include_router(chat_router)
    app.include_router(chat_langchain_router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"message": settings.app_name}

    return app


app = create_app()
