from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import torch
from ..models.neural import NeuralNetworkRequest
from ..services.neural_service import NeuralService
from ..auth.dependencies import get_current_user

router = APIRouter()
neural_service = NeuralService()

@router.post("/train")
async def train_neural_network(
    request: NeuralNetworkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Train quantum neural network."""
    try:
        result = neural_service.train(
            input_data=request.input_data,
            model_config=request.model_config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
async def neural_network_predict(
    request: NeuralNetworkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Make predictions using neural network."""
    try:
        result = neural_service.predict(
            input_data=request.input_data,
            model_config=request.model_config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_network(
    request: NeuralNetworkRequest,
    current_user: dict = Depends(get_current_user)
):
    """Optimize neural network architecture."""
    try:
        result = neural_service.optimize_network(
            input_data=request.input_data,
            model_config=request.model_config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models(
    current_user: dict = Depends(get_current_user)
):
    """List available neural network models."""
    try:
        models = neural_service.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
