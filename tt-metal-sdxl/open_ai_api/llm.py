# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from fastapi import APIRouter, Depends

from model_services.base_model import BaseModel
from resolver.model_resolver import model_resolver

router = APIRouter()

@router.post('/completions')
def completions(service: BaseModel = Depends(model_resolver)):
    return service.completions()
