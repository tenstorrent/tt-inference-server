import asyncio

from httpcore import request
from domain.text_embedding_request import TextEmbeddingRequest
from model_services.base_service import BaseService

class LLMService(BaseService):

    def __init__(self):
        super().__init__()

    async def process_request(self, request: TextEmbeddingRequest):
        if not isinstance(request.input, list):
            return await super().process_request(request)

        individual_requests = []
        for i in range(len(request.input)):
            
            field_values = request.model_dump()
            new_request = type(request)(**field_values)

            new_request.input = request.input[i]

            individual_requests.append(new_request)
        
        # Create tasks using a regular loop instead of list comprehension
        tasks = []
        for req in individual_requests:
            tasks.append(super().process_request(req))
        
        # Gather results
        results = await asyncio.gather(*tasks)    
        
        return results
