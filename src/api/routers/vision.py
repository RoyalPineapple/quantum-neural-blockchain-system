from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import torch
from ..models.vision import VisionRequest
from ..services.vision_service import VisionService
from ..auth.dependencies import get_current_user

router = APIRouter()
vision_service = VisionService()

@router.post("/process")
async def process_vision(
    request: VisionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process image using quantum vision system."""
    try:
        result = vision_service.process_image(
            image_data=request.image_data,
            task=request.task,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect")
async def detect_objects(
    request: VisionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Detect objects in image."""
    try:
        result = vision_service.detect_objects(
            image_data=request.image_data,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/segment")
async def segment_image(
    request: VisionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Perform image segmentation."""
    try:
        result = vision_service.segment_image(
            image_data=request.image_data,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reconstruct")
async def reconstruct_image(
    request: VisionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Reconstruct image using quantum features."""
    try:
        result = vision_service.reconstruct_image(
            image_data=request.image_data,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_vision_models(
    current_user: dict = Depends(get_current_user)
):
    """List available vision models."""
    try:
        models = vision_service.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
