from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel

class FineTunedModelItem(BaseModel):
    id: str
    tag: str
    base_model_id: str
    created_at: Optional[datetime] = None

class FineTunedModelsResponse(BaseModel):
    base_model_id: str
    data: List[FineTunedModelItem]