from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from ..models.nlp import NLPRequest
from ..services.nlp_service import NLPService
from ..auth.dependencies import get_current_user

router = APIRouter()
nlp_service = NLPService()

@router.post("/process")
async def process_text(
    request: NLPRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process text using quantum NLP system."""
    try:
        result = nlp_service.process_text(
            text=request.text,
            task=request.task,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_text(
    request: NLPRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate text using quantum language model."""
    try:
        result = nlp_service.generate_text(
            prompt=request.text,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate")
async def translate_text(
    request: NLPRequest,
    current_user: dict = Depends(get_current_user)
):
    """Translate text using quantum translation model."""
    try:
        result = nlp_service.translate_text(
            text=request.text,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_text(
    request: NLPRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze text sentiment and features."""
    try:
        result = nlp_service.analyze_text(
            text=request.text,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_nlp_models(
    current_user: dict = Depends(get_current_user)
):
    """List available NLP models."""
    try:
        models = nlp_service.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
