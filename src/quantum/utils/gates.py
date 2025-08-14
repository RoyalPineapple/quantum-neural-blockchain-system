import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

class GateType(Enum):
    """Types of quantum gates."""
    SINGLE_QUBIT = "single_qubit"
    TWO_QUBIT = "two_qubit"
    THREE_QUBIT = "three_qubit"
    CUSTOM = "custom"

@dataclass
class GateProperties:
    """Properties of a quantum gate."""
    type: GateType
    is_hermitian: bool
    is_unitary: bool
    is_clifford: bool
    required_qubits: int
    decomposable: bool
    noise_resilience: float  # 0 to 1, higher is better

class QuantumGate:
    """
    Implementation of quantum gates with advanced features including:
    - Parameterized gates
    - Noise-aware operations
    - Gate fusion optimization
    - Automatic decomposition
    - Error tracking
    """
    
    def __init__(
        self,
        matrix: np.ndarray,
        properties: GateProperties,
        name: str,
        params: Optional[List[float]] = None,
        noise_model: Optional[dict] = None
    ):
        """
        Initialize quantum gate.
        
        Args:
            matrix: Gate unitary matrix
            properties: Gate properties
            name: Gate name
            params: Optional parameters for parameterized gates
            noise_model: Optional noise model
        """
        self.matrix = matrix
        self.properties = properties
        self.name = name
        self.params = params or []
        self.noise_model = noise_model or {}
        
        # Validate matrix properties
        self._validate_matrix()
        
        # Initialize error tracking
        self.error_stats = {
            "total_applications": 0,
            "error_occurrences": 0,
            "average_fidelity": 1.0
        }
        
    def _validate_matrix(self) -> None:
        """Validate quantum gate matrix properties."""
        # Check matrix shape
        n_qubits = int(np.log2(self.matrix.shape[0]))
        if self.matrix.shape != (2**n_qubits, 2**n_qubits):
            raise ValueError("Invalid matrix dimensions")
            
        # Check unitarity if claimed
        if self.properties.is_unitary:
            identity = np.eye(self.matrix.shape[0])
            if not np.allclose(
                self.matrix @ self.matrix.conj().T,
                identity,
                atol=1e-10
            ):
                raise ValueError("Matrix is not unitary")
                
        # Check hermiticity if claimed
        if self.properties.is_hermitian:
            if not np.allclose(
                self.matrix,
                self.matrix.conj().T,
                atol=1e-10
            ):
                raise ValueError("Matrix is not hermitian")
                
    def apply(
        self,
        state: np.ndarray,
        qubits: List[int],
        noise_aware: bool = True
    ) -> np.ndarray:
        """
        Apply gate to quantum state.
        
        Args:
            state: Quantum state vector
            qubits: Qubits to apply gate to
            noise_aware: Whether to consider noise in operation
            
        Returns:
            New quantum state after gate application
        """
        # Validate number of qubits
        if len(qubits) != self.properties.required_qubits:
            raise ValueError(
                f"Gate {self.name} requires {self.properties.required_qubits} qubits, "
                f"got {len(qubits)}"
            )
            
        # Apply gate operation
        new_state = self._apply_operation(state, qubits)
        
        # Apply noise if enabled
        if noise_aware and self.noise_model:
            new_state = self._apply_noise(new_state, qubits)
            
        # Update error statistics
        self._update_error_stats(state, new_state, qubits)
        
        return new_state
        
    def _apply_operation(
        self,
        state: np.ndarray,
        qubits: List[int]
    ) -> np.ndarray:
        """Apply ideal gate operation."""
        n_qubits = int(np.log2(len(state)))
        
        # Create full operation matrix
        if self.properties.type == GateType.SINGLE_QUBIT:
            operation = self._extend_single_qubit(n_qubits, qubits[0])
        elif self.properties.type == GateType.TWO_QUBIT:
            operation = self._extend_two_qubit(n_qubits, qubits[0], qubits[1])
        else:
            operation = self._extend_multi_qubit(n_qubits, qubits)
            
        return operation @ state
        
    def _apply_noise(
        self,
        state: np.ndarray,
        qubits: List[int]
    ) -> np.ndarray:
        """Apply noise effects to state."""
        noisy_state = state.copy()
        
        for qubit in qubits:
            # Apply depolarizing noise
            if "depolarizing" in self.noise_model:
                p = self.noise_model["depolarizing"]
                noisy_state = self._apply_depolarizing_noise(noisy_state, qubit, p)
                
            # Apply amplitude damping
            if "amplitude_damping" in self.noise_model:
                gamma = self.noise_model["amplitude_damping"]
                noisy_state = self._apply_amplitude_damping(noisy_state, qubit, gamma)
                
            # Apply phase damping
            if "phase_damping" in self.noise_model:
                lambda_param = self.noise_model["phase_damping"]
                noisy_state = self._apply_phase_damping(noisy_state, qubit, lambda_param)
                
        return noisy_state
        
    def _extend_single_qubit(
        self,
        n_qubits: int,
        target: int
    ) -> np.ndarray:
        """Extend single-qubit gate to full system."""
        # Create identity matrices for other qubits
        left_qubits = target
        right_qubits = n_qubits - target - 1
        
        # Tensor product: I⊗...⊗I⊗U⊗I⊗...⊗I
        operation = np.eye(1)
        
        # Add identity matrices to the left
        for _ in range(left_qubits):
            operation = np.kron(np.eye(2), operation)
            
        # Add gate matrix
        operation = np.kron(self.matrix, operation)
        
        # Add identity matrices to the right
        for _ in range(right_qubits):
            operation = np.kron(np.eye(2), operation)
            
        return operation
        
    def _extend_two_qubit(
        self,
        n_qubits: int,
        control: int,
        target: int
    ) -> np.ndarray:
        """Extend two-qubit gate to full system."""
        if control == target:
            raise ValueError("Control and target qubits must be different")
            
        # Sort qubits to handle different orderings
        q1, q2 = min(control, target), max(control, target)
        
        # Calculate dimensions
        dim_left = 2**q1
        dim_middle = 2**(q2 - q1 - 1)
        dim_right = 2**(n_qubits - q2 - 1)
        
        # Create extended matrix
        extended = np.eye(1)
        
        # Add identity matrices for qubits before first gate qubit
        for _ in range(q1):
            extended = np.kron(np.eye(2), extended)
            
        # Add gate matrix
        if control < target:
            extended = np.kron(self.matrix, extended)
        else:
            # Swap matrix if control > target
            perm = np.array([0,2,1,3]).reshape(4,1)
            swapped = self.matrix[perm, perm.T]
            extended = np.kron(swapped, extended)
            
        # Add identity matrices for remaining qubits
        for _ in range(n_qubits - q2 - 1):
            extended = np.kron(np.eye(2), extended)
            
        return extended
        
    def _extend_multi_qubit(
        self,
        n_qubits: int,
        qubits: List[int]
    ) -> np.ndarray:
        """Extend multi-qubit gate to full system."""
        # Sort qubits
        sorted_qubits = sorted(qubits)
        
        # Create permutation to move target qubits together
        perm = list(range(n_qubits))
        for i, qubit in enumerate(sorted_qubits):
            perm.remove(qubit)
            perm.insert(i, qubit)
            
        # Create permutation matrices
        perm_matrix = self._create_permutation_matrix(n_qubits, perm)
        inv_perm_matrix = perm_matrix.T
        
        # Extend gate matrix to intermediate qubits
        extended = self.matrix
        for i in range(n_qubits - len(qubits)):
            extended = np.kron(extended, np.eye(2))
            
        # Apply permutations
        return inv_perm_matrix @ extended @ perm_matrix
        
    def _create_permutation_matrix(
        self,
        n_qubits: int,
        perm: List[int]
    ) -> np.ndarray:
        """Create permutation matrix for qubit reordering."""
        dim = 2**n_qubits
        matrix = np.zeros((dim, dim))
        
        for i in range(dim):
            # Convert index to binary and permute bits
            binary = format(i, f'0{n_qubits}b')
            permuted = ''.join(binary[p] for p in perm)
            j = int(permuted, 2)
            matrix[j, i] = 1
            
        return matrix
        
    def _apply_depolarizing_noise(
        self,
        state: np.ndarray,
        qubit: int,
        p: float
    ) -> np.ndarray:
        """Apply depolarizing noise to qubit."""
        if p == 0:
            return state
            
        # Pauli matrices
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        
        # Apply each Pauli error with probability p/3
        noisy_state = (1 - p) * state
        
        for pauli in [X, Y, Z]:
            pauli_operation = self._extend_single_qubit(
                int(np.log2(len(state))),
                qubit
            )
            noisy_state += (p/3) * (pauli_operation @ state)
            
        return noisy_state
        
    def _apply_amplitude_damping(
        self,
        state: np.ndarray,
        qubit: int,
        gamma: float
    ) -> np.ndarray:
        """Apply amplitude damping noise."""
        if gamma == 0:
            return state
            
        # Kraus operators for amplitude damping
        K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        # Extend to full system
        n_qubits = int(np.log2(len(state)))
        K0_full = self._extend_single_qubit(n_qubits, qubit)
        K1_full = self._extend_single_qubit(n_qubits, qubit)
        
        # Apply Kraus operators
        return (K0_full @ state) + (K1_full @ state)
        
    def _apply_phase_damping(
        self,
        state: np.ndarray,
        qubit: int,
        lambda_param: float
    ) -> np.ndarray:
        """Apply phase damping noise."""
        if lambda_param == 0:
            return state
            
        # Kraus operators for phase damping
        K0 = np.array([[1, 0], [0, np.sqrt(1-lambda_param)]])
        K1 = np.array([[0, 0], [0, np.sqrt(lambda_param)]])
        
        # Extend to full system
        n_qubits = int(np.log2(len(state)))
        K0_full = self._extend_single_qubit(n_qubits, qubit)
        K1_full = self._extend_single_qubit(n_qubits, qubit)
        
        # Apply Kraus operators
        return (K0_full @ state) + (K1_full @ state)
        
    def _update_error_stats(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        qubits: List[int]
    ) -> None:
        """Update gate error statistics."""
        self.error_stats["total_applications"] += 1
        
        # Calculate state fidelity
        fidelity = np.abs(np.vdot(initial_state, final_state))**2
        
        # Update running average
        n = self.error_stats["total_applications"]
        old_avg = self.error_stats["average_fidelity"]
        self.error_stats["average_fidelity"] = (
            (n - 1) * old_avg + fidelity
        ) / n
        
        # Count error occurrence
        if fidelity < 0.99:  # Threshold for counting as error
            self.error_stats["error_occurrences"] += 1
            
    @classmethod
    def hadamard(cls) -> 'QuantumGate':
        """Create Hadamard gate."""
        matrix = np.array([
            [1, 1],
            [1, -1]
        ]) / np.sqrt(2)
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=1,
            decomposable=False,
            noise_resilience=0.8
        )
        
        return cls(matrix, properties, "H")
        
    @classmethod
    def pauli_x(cls) -> 'QuantumGate':
        """Create Pauli-X (NOT) gate."""
        matrix = np.array([
            [0, 1],
            [1, 0]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=1,
            decomposable=False,
            noise_resilience=0.9
        )
        
        return cls(matrix, properties, "X")
        
    @classmethod
    def pauli_y(cls) -> 'QuantumGate':
        """Create Pauli-Y gate."""
        matrix = np.array([
            [0, -1j],
            [1j, 0]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=1,
            decomposable=False,
            noise_resilience=0.9
        )
        
        return cls(matrix, properties, "Y")
        
    @classmethod
    def pauli_z(cls) -> 'QuantumGate':
        """Create Pauli-Z gate."""
        matrix = np.array([
            [1, 0],
            [0, -1]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=1,
            decomposable=False,
            noise_resilience=0.95
        )
        
        return cls(matrix, properties, "Z")
        
    @classmethod
    def rx(cls, theta: float) -> 'QuantumGate':
        """Create rotation around X axis."""
        matrix = np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=False,
            is_unitary=True,
            is_clifford=False,
            required_qubits=1,
            decomposable=True,
            noise_resilience=0.7
        )
        
        return cls(matrix, properties, "RX", params=[theta])
        
    @classmethod
    def ry(cls, theta: float) -> 'QuantumGate':
        """Create rotation around Y axis."""
        matrix = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=False,
            is_unitary=True,
            is_clifford=False,
            required_qubits=1,
            decomposable=True,
            noise_resilience=0.7
        )
        
        return cls(matrix, properties, "RY", params=[theta])
        
    @classmethod
    def rz(cls, theta: float) -> 'QuantumGate':
        """Create rotation around Z axis."""
        matrix = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ])
        
        properties = GateProperties(
            type=GateType.SINGLE_QUBIT,
            is_hermitian=False,
            is_unitary=True,
            is_clifford=False,
            required_qubits=1,
            decomposable=True,
            noise_resilience=0.75
        )
        
        return cls(matrix, properties, "RZ", params=[theta])
        
    @classmethod
    def cnot(
        cls,
        strength: float = 1.0,
        noise_model: Optional[dict] = None
    ) -> 'QuantumGate':
        """Create CNOT (controlled-X) gate."""
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        # Apply strength parameter
        if strength != 1.0:
            # Interpolate between identity and CNOT
            identity = np.eye(4)
            matrix = (1 - strength) * identity + strength * matrix
            
        properties = GateProperties(
            type=GateType.TWO_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=2,
            decomposable=True,
            noise_resilience=0.6
        )
        
        return cls(
            matrix,
            properties,
            "CNOT",
            params=[strength],
            noise_model=noise_model
        )
        
    @classmethod
    def cz(cls) -> 'QuantumGate':
        """Create controlled-Z gate."""
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ])
        
        properties = GateProperties(
            type=GateType.TWO_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=2,
            decomposable=True,
            noise_resilience=0.65
        )
        
        return cls(matrix, properties, "CZ")
        
    @classmethod
    def swap(cls) -> 'QuantumGate':
        """Create SWAP gate."""
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        
        properties = GateProperties(
            type=GateType.TWO_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=True,
            required_qubits=2,
            decomposable=True,
            noise_resilience=0.5
        )
        
        return cls(matrix, properties, "SWAP")
        
    @classmethod
    def toffoli(cls) -> 'QuantumGate':
        """Create Toffoli (CCNOT) gate."""
        matrix = np.zeros((8, 8))
        matrix[:6, :6] = np.eye(6)
        matrix[6:, 6:] = np.array([[0, 1], [1, 0]])
        
        properties = GateProperties(
            type=GateType.THREE_QUBIT,
            is_hermitian=True,
            is_unitary=True,
            is_clifford=False,
            required_qubits=3,
            decomposable=True,
            noise_resilience=0.4
        )
        
        return cls(matrix, properties, "TOFFOLI")
        
    @classmethod
    def custom(
        cls,
        matrix: np.ndarray,
        name: str,
        required_qubits: int,
        is_unitary: bool = True,
        is_hermitian: bool = False,
        noise_model: Optional[dict] = None
    ) -> 'QuantumGate':
        """Create custom quantum gate."""
        properties = GateProperties(
            type=GateType.CUSTOM,
            is_hermitian=is_hermitian,
            is_unitary=is_unitary,
            is_clifford=False,
            required_qubits=required_qubits,
            decomposable=True,
            noise_resilience=0.5
        )
        
        return cls(matrix, properties, name, noise_model=noise_model)
        
    def is_compatible(self, other: 'QuantumGate') -> bool:
        """Check if gates can be combined/simplified."""
        return (
            self.properties.type == other.properties.type and
            self.properties.required_qubits == other.properties.required_qubits
        )
        
    def combine(self, other: 'QuantumGate') -> 'QuantumGate':
        """Combine two compatible gates."""
        if not self.is_compatible(other):
            raise ValueError("Gates are not compatible for combination")
            
        # Multiply matrices
        combined_matrix = self.matrix @ other.matrix
        
        # Determine combined properties
        properties = GateProperties(
            type=self.properties.type,
            is_hermitian=False,  # Need to check explicitly
            is_unitary=self.properties.is_unitary and other.properties.is_unitary,
            is_clifford=self.properties.is_clifford and other.properties.is_clifford,
            required_qubits=self.properties.required_qubits,
            decomposable=True,
            noise_resilience=min(
                self.properties.noise_resilience,
                other.properties.noise_resilience
            )
        )
        
        return QuantumGate(
            combined_matrix,
            properties,
            f"{self.name}+{other.name}"
        )
        
    def decompose(self) -> List['QuantumGate']:
        """Decompose gate into simpler gates."""
        if not self.properties.decomposable:
            return [self]
            
        if self.properties.type == GateType.TWO_QUBIT:
            return self._decompose_two_qubit()
        elif self.properties.type == GateType.THREE_QUBIT:
            return self._decompose_three_qubit()
        else:
            return [self]
            
    def _decompose_two_qubit(self) -> List['QuantumGate']:
        """Decompose two-qubit gate into single-qubit gates and CNOTs."""
        # Use Quantum Shannon Decomposition
        # This is a placeholder - actual implementation would use
        # proper decomposition algorithms
        return [self]
        
    def _decompose_three_qubit(self) -> List['QuantumGate']:
        """Decompose three-qubit gate into two-qubit gates."""
        # Placeholder for three-qubit gate decomposition
        return [self]
        
    def get_error_rate(self) -> float:
        """Get gate error rate from statistics."""
        if self.error_stats["total_applications"] == 0:
            return 0.0
            
        return (
            self.error_stats["error_occurrences"] /
            self.error_stats["total_applications"]
        )
        
    def __str__(self) -> str:
        """String representation of gate."""
        return f"{self.name}({', '.join(map(str, self.params))})" if self.params else self.name
