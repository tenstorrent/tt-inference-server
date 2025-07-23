from fastapi import APIRouter, Depends, File, Response, Security
from fastapi.concurrency import run_in_threadpool
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_model import BaseModel
from resolver.model_resolver import model_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/generations')
async def generateImage(imageGenerateRequest: ImageGenerateRequest, service: BaseModel = Depends(model_resolver), api_key: str = Security(get_api_key)):
    try:
        result = await service.processImage(imageGenerateRequest)
    except Exception as e:
        return Response(status_code=500, content=str(e))
    return Response(content=result, media_type="image/png")

@router.get('/tt-liveness')
def liveness(service: BaseModel = Depends(model_resolver)):
    return {'status': 'alive', 'is_ready': service.checkIsModelReady()}