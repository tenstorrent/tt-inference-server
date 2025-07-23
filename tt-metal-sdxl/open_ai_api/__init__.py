from fastapi import APIRouter

api_router = APIRouter()

from open_ai_api import image, llm

api_router.include_router(image.router, prefix='/image', tags=['Image processing'])
api_router.include_router(llm.router, prefix='', tags=['Language processing'])
