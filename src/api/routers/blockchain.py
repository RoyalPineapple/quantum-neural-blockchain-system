from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from ..models.blockchain import BlockchainRequest
from ..services.blockchain_service import BlockchainService
from ..auth.dependencies import get_current_user

router = APIRouter()
blockchain_service = BlockchainService()

@router.post("/transaction")
async def create_transaction(
    request: BlockchainRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create new blockchain transaction."""
    try:
        result = blockchain_service.create_transaction(
            sender=request.sender,
            receiver=request.receiver,
            amount=request.amount,
            data=request.transaction_data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_blockchain_status(
    current_user: dict = Depends(get_current_user)
):
    """Get blockchain status."""
    try:
        status = blockchain_service.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/mine")
async def mine_block(
    current_user: dict = Depends(get_current_user)
):
    """Mine new block."""
    try:
        result = blockchain_service.mine_block()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/verify")
async def verify_chain(
    current_user: dict = Depends(get_current_user)
):
    """Verify blockchain integrity."""
    try:
        result = blockchain_service.verify_chain()
        return {"valid": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/balance/{address}")
async def get_balance(
    address: str,
    current_user: dict = Depends(get_current_user)
):
    """Get balance for address."""
    try:
        balance = blockchain_service.get_balance(address)
        return {"address": address, "balance": balance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
