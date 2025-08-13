from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import torch
from ..models.robotics import RoboticsRequest
from ..services.robotics_service import RoboticsService
from ..auth.dependencies import get_current_user

router = APIRouter()
robotics_service = RoboticsService()

@router.post("/plan")
async def plan_trajectory(
    request: RoboticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Plan robot trajectory using quantum optimization."""
    try:
        result = robotics_service.plan_trajectory(
            state=request.state,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/control")
async def execute_control(
    request: RoboticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute robot control action."""
    try:
        result = robotics_service.execute_control(
            state=request.state,
            action=request.action,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/coordinate")
async def coordinate_swarm(
    request: RoboticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Coordinate robot swarm behavior."""
    try:
        result = robotics_service.coordinate_swarm(
            states=request.state,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state")
async def get_robot_state(
    current_user: dict = Depends(get_current_user)
):
    """Get current robot state."""
    try:
        state = robotics_service.get_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_control(
    request: RoboticsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Optimize robot control parameters."""
    try:
        result = robotics_service.optimize_control(
            state=request.state,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
