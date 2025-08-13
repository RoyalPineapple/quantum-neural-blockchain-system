from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import jwt
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Neural Blockchain System API",
    description="API for quantum computing, neural networks, blockchain, and more",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "your-secret-key-here"  # In production, use secure key
ALGORITHM = "HS256"

# Request/Response Models
class QuantumCircuitRequest(BaseModel):
    n_qubits: int
    operations: List[Dict[str, Any]]
    params: Optional[Dict[str, Any]] = None

class QuantumStateRequest(BaseModel):
    state_vector: List[complex]
    n_qubits: int

class NeuralNetworkRequest(BaseModel):
    input_data: List[List[float]]
    model_config: Dict[str, Any]

class BlockchainRequest(BaseModel):
    transaction_data: Dict[str, Any]
    sender: str
    receiver: str
    amount: float

class VisionRequest(BaseModel):
    image_data: List[List[List[float]]]
    task: str
    params: Optional[Dict[str, Any]] = None

class NLPRequest(BaseModel):
    text: str
    task: str
    params: Optional[Dict[str, Any]] = None

class RoboticsRequest(BaseModel):
    state: List[float]
    action: str
    params: Optional[Dict[str, Any]] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = {"username": username}
    except jwt.JWTError:
        raise credentials_exception
    return token_data

# API Routes

# Authentication
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, verify credentials against database
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Quantum Computing Routes
@app.post("/quantum/circuit")
async def execute_quantum_circuit(
    request: QuantumCircuitRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.quantum.core.quantum_register import QuantumRegister
        from src.quantum.utils.gates import QuantumGate
        
        # Initialize quantum register
        qreg = QuantumRegister(request.n_qubits)
        
        # Execute operations
        for op in request.operations:
            qreg.apply_gate(
                QuantumGate(op['gate_type'], op.get('params', {})),
                op['qubits']
            )
        
        # Get final state
        final_state = qreg.get_state()
        
        return {
            "state": final_state.tolist(),
            "measurements": qreg.measure()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/state")
async def analyze_quantum_state(
    request: QuantumStateRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        state_vector = np.array(request.state_vector)
        
        # Calculate quantum properties
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        return {
            "entropy": entropy,
            "purity": np.abs(np.trace(density_matrix)),
            "probability_distribution": np.abs(state_vector)**2
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Neural Network Routes
@app.post("/neural/train")
async def train_neural_network(
    request: NeuralNetworkRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.neural.core.quantum_neural_network import QuantumNeuralNetwork
        
        # Convert input data to tensor
        input_data = torch.tensor(request.input_data)
        
        # Initialize and train network
        model = QuantumNeuralNetwork(**request.model_config)
        result = model.train(input_data)
        
        return {
            "training_loss": result['loss'],
            "accuracy": result['accuracy'],
            "quantum_metrics": result['quantum_metrics']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neural/predict")
async def neural_network_predict(
    request: NeuralNetworkRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.neural.core.quantum_neural_network import QuantumNeuralNetwork
        
        # Convert input data to tensor
        input_data = torch.tensor(request.input_data)
        
        # Initialize and predict
        model = QuantumNeuralNetwork(**request.model_config)
        predictions = model(input_data)
        
        return {
            "predictions": predictions.tolist(),
            "confidence": torch.softmax(predictions, dim=1).tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Blockchain Routes
@app.post("/blockchain/transaction")
async def create_transaction(
    request: BlockchainRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.blockchain.core.blockchain import QuantumBlockchain
        
        # Initialize blockchain
        blockchain = QuantumBlockchain()
        
        # Create transaction
        transaction = {
            "sender": request.sender,
            "receiver": request.receiver,
            "amount": request.amount,
            "timestamp": datetime.utcnow().timestamp(),
            "data": request.transaction_data
        }
        
        # Add transaction
        block_index = blockchain.add_transaction(transaction)
        
        return {
            "transaction_hash": blockchain.get_last_block().hash,
            "block_index": block_index,
            "status": "pending"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/blockchain/status")
async def get_blockchain_status(
    current_user: User = Depends(get_current_user)
):
    try:
        from src.blockchain.core.blockchain import QuantumBlockchain
        
        blockchain = QuantumBlockchain()
        
        return {
            "chain_length": len(blockchain.chain),
            "last_block_hash": blockchain.get_last_block().hash,
            "pending_transactions": len(blockchain.pending_transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Vision Routes
@app.post("/vision/process")
async def process_vision(
    request: VisionRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.vision.core.vision_system import QuantumVisionSystem
        
        # Convert input to tensor
        image_data = torch.tensor(request.image_data)
        
        # Initialize vision system
        vision_system = QuantumVisionSystem()
        
        # Process image
        result = vision_system.process_image(
            image_data,
            task=request.task,
            **request.params or {}
        )
        
        return {
            "processed_data": result['data'].tolist(),
            "detected_features": result['features'],
            "quantum_analysis": result['quantum_analysis']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NLP Routes
@app.post("/nlp/process")
async def process_text(
    request: NLPRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.nlp.core.nlp_system import QuantumNLPSystem
        
        # Initialize NLP system
        nlp_system = QuantumNLPSystem()
        
        # Process text
        result = nlp_system.process_text(
            request.text,
            task=request.task,
            **request.params or {}
        )
        
        return {
            "processed_text": result['text'],
            "analysis": result['analysis'],
            "quantum_features": result['quantum_features']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Robotics Routes
@app.post("/robotics/control")
async def control_robotics(
    request: RoboticsRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        from src.robotics.core.robotics_system import QuantumRoboticsSystem
        
        # Initialize robotics system
        robotics_system = QuantumRoboticsSystem()
        
        # Execute control action
        result = robotics_system.execute_action(
            state=torch.tensor(request.state),
            action=request.action,
            **request.params or {}
        )
        
        return {
            "next_state": result['next_state'].tolist(),
            "action_taken": result['action'],
            "quantum_control": result['quantum_control']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System Status Route
@app.get("/system/status")
async def get_system_status(
    current_user: User = Depends(get_current_user)
):
    try:
        return {
            "status": "operational",
            "components": {
                "quantum": "online",
                "neural": "online",
                "blockchain": "online",
                "vision": "online",
                "nlp": "online",
                "robotics": "online"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
