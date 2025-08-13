import numpy as np
from typing import List, Optional, Dict, Any
from enum import Enum

class GateType(Enum):
    """Enumeration of supported quantum gates."""
    H = "Hadamard"
    X = "Pauli-X"
    Y = "Pauli-Y"
    Z = "Pauli-Z"
    CNOT = "Controlled-NOT"
    SWAP = "SWAP"
    T = "T-gate"
    S = "S-gate"
    Rx = "Rotation-X"
    Ry = "Rotation-Y"
    Rz = "Rotation-Z"
    U = "Universal"

class QuantumGate:
    """
    Implementation of quantum gates with support for parameterized operations
    and efficient matrix representations.
    """
    
    def __init__(self, gate_type: GateType, params: Optional[Dict[str, float]] = None):
        """
        Initialize a quantum gate.
        
        Args:
            gate_type: Type of quantum gate
            params: Optional parameters for parameterized gates
        """
        self.gate_type = gate_type
        self.params = params or {}
        self.matrix = self._build_matrix()
        
    def _build_matrix(self) -> np.ndarray:
        """
        Build the matrix representation of the quantum gate.
        
        Returns:
            Complex numpy array representing gate operation
        """
        if self.gate_type == GateType.H:
            return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
            
        elif self.gate_type == GateType.X:
            return np.array([[0, 1], [1, 0]], dtype=np.complex128)
            
        elif self.gate_type == GateType.Y:
            return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            
        elif self.gate_type == GateType.Z:
            return np.array([[1, 0], [0, -1]], dtype=np.complex128)
            
        elif self.gate_type == GateType.CNOT:
            return np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.SWAP:
            return np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.T:
            return np.array([[1, 0],
                           [0, np.exp(1j * np.pi/4)]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.S:
            return np.array([[1, 0],
                           [0, 1j]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.Rx:
            theta = self.params.get('theta', 0.0)
            c = np.cos(theta/2)
            s = -1j * np.sin(theta/2)
            return np.array([[c, s],
                           [s, c]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.Ry:
            theta = self.params.get('theta', 0.0)
            c = np.cos(theta/2)
            s = np.sin(theta/2)
            return np.array([[c, -s],
                           [s, c]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.Rz:
            theta = self.params.get('theta', 0.0)
            return np.array([[np.exp(-1j * theta/2), 0],
                           [0, np.exp(1j * theta/2)]], dtype=np.complex128)
                           
        elif self.gate_type == GateType.U:
            theta = self.params.get('theta', 0.0)
            phi = self.params.get('phi', 0.0)
            lambda_ = self.params.get('lambda', 0.0)
            
            return np.array([
                [np.cos(theta/2), -np.exp(1j*lambda_)*np.sin(theta/2)],
                [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lambda_))*np.cos(theta/2)]
            ], dtype=np.complex128)
            
        else:
            raise ValueError(f"Unsupported gate type: {self.gate_type}")
    
    def apply(self, state: np.ndarray, qubits: List[int]) -> np.ndarray:
        """
        Apply the quantum gate to a quantum state.
        
        Args:
            state: Quantum state vector to apply gate to
            qubits: List of qubit indices to apply gate to
            
        Returns:
            Updated quantum state after gate application
        """
        n_qubits = int(np.log2(len(state)))
        
        # Validate number of qubits matches gate
        required_qubits = 2 if self.gate_type in [GateType.CNOT, GateType.SWAP] else 1
        if len(qubits) != required_qubits:
            raise ValueError(f"{self.gate_type} requires {required_qubits} qubit(s)")
            
        # Build full operation matrix
        if required_qubits == 1:
            op = self._build_single_qubit_operation(n_qubits, qubits[0])
        else:
            op = self._build_two_qubit_operation(n_qubits, qubits[0], qubits[1])
            
        # Apply operation
        return np.dot(op, state)
    
    def _build_single_qubit_operation(self, n_qubits: int, target: int) -> np.ndarray:
        """
        Build operation matrix for single-qubit gate.
        
        Args:
            n_qubits: Total number of qubits in system
            target: Target qubit index
            
        Returns:
            Operation matrix for full system
        """
        # Identity matrices for other qubits
        id_before = np.eye(2**target, dtype=np.complex128)
        id_after = np.eye(2**(n_qubits-target-1), dtype=np.complex128)
        
        # Kronecker product to build full operation
        return np.kron(np.kron(id_before, self.matrix), id_after)
    
    def _build_two_qubit_operation(self, n_qubits: int, control: int, target: int) -> np.ndarray:
        """
        Build operation matrix for two-qubit gate.
        
        Args:
            n_qubits: Total number of qubits in system
            control: Control qubit index
            target: Target qubit index
            
        Returns:
            Operation matrix for full system
        """
        # Ensure control comes before target
        if control > target:
            control, target = target, control
            if self.gate_type == GateType.CNOT:
                # Adjust CNOT matrix for swapped qubits
                self.matrix = np.array([[1, 0, 0, 0],
                                      [0, 0, 0, 1],
                                      [0, 0, 1, 0],
                                      [0, 1, 0, 0]], dtype=np.complex128)
        
        # Identity matrices for other qubits
        id_before = np.eye(2**control, dtype=np.complex128)
        id_between = np.eye(2**(target-control-1), dtype=np.complex128)
        id_after = np.eye(2**(n_qubits-target-1), dtype=np.complex128)
        
        # Kronecker product to build full operation
        return np.kron(np.kron(np.kron(id_before, self.matrix), id_between), id_after)
