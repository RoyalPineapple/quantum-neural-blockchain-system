from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
import torch
from ..models.quantum import QuantumCircuitRequest, QuantumStateRequest
from ..services.quantum_service import QuantumService
from ..auth.dependencies import get_current_user

router = APIRouter()
quantum_service = QuantumService()

@router.post("/circuit")
async def execute_quantum_circuit(
    request: QuantumCircuitRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute quantum circuit operations."""
    try:
        result = quantum_service.execute_circuit(
            n_qubits=request.n_qubits,
            operations=request.operations,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/state")
async def analyze_quantum_state(
    request: QuantumStateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze quantum state properties."""
    try:
        result = quantum_service.analyze_state(
            state_vector=request.state_vector,
            n_qubits=request.n_qubits
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/gates")
async def list_quantum_gates(
    current_user: dict = Depends(get_current_user)
):
    """List available quantum gates."""
    try:
        gates = quantum_service.list_gates()
        return {"gates": gates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_circuit(
    request: QuantumCircuitRequest,
    current_user: dict = Depends(get_current_user)
):
    """Optimize quantum circuit."""
    try:
        result = quantum_service.optimize_circuit(
            n_qubits=request.n_qubits,
            operations=request.operations,
            params=request.params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
