from model_services.base_model import BaseModel
from model_services.sdxl_service import SDXLService
from config.settings import settings
# from model_services.task_worker import TaskWorker

# model and worker are singleton
current_model_holder = None

def model_resolver() -> BaseModel:
    global current_model_holder
    model_in_use = settings.model_in_use
    if model_in_use == "SDXL":
        if (current_model_holder is None):
            current_model_holder = SDXLService()
        return current_model_holder    
    return BaseModel()