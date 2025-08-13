import numpy as np
from typing import List, Optional, Tuple, Dict
from ..utils.error_correction import ErrorCorrector
from ..utils.gates import QuantumGate

class QuantumRegister:
    """
    A quantum register implementation supporting large-scale quantum computation
    with error correction and efficient state management.
    """
    
    def __init__(self, n_qubits: int, error_threshold: float = 0.001):
        """
        Initialize a quantum register with specified number of qubits.
        
        Args:
            n_qubits: Number of qubits in the register
            error_threshold: Threshold for error correction
        """
        if not 1 <= n_qubits <= 1024:
            raise ValueError("Number of qubits must be between 1 and 1024")
            
        self.n_qubits = n_qubits
        self.error_threshold = error_threshold
        
        # Initialize quantum state to |0>^⊗n
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        
        # Initialize error corrector
        self.error_corrector = ErrorCorrector(error_threshold)
        
        # Track gate history for optimization
        self.gate_history: List[Tuple[QuantumGate, List[int]]] = []
        
        # Measurement results cache
        self.measurements: Dict[int, int] = {}
        
    def apply_gate(self, gate: QuantumGate, qubits: List[int]) -> None:
        """
        Apply a quantum gate to specified qubits.
        
        Args:
            gate: Quantum gate to apply
            qubits: List of qubit indices to apply gate to
        """
        # Validate qubit indices
        if not all(0 <= q < self.n_qubits for q in qubits):
            raise ValueError("Invalid qubit indices")
            
        # Apply gate operation
        self.state = gate.apply(self.state, qubits)
        
        # Record gate application
        self.gate_history.append((gate, qubits))
        
        # Perform error correction if needed
        if self.error_corrector.needs_correction(self.state):
            self.state = self.error_corrector.correct(self.state)
    
    def measure(self, qubits: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Measure specified qubits or all qubits if none specified.
        
        Args:
            qubits: Optional list of qubit indices to measure
            
        Returns:
            Dictionary mapping qubit indices to measured values (0 or 1)
        """
        qubits = qubits or list(range(self.n_qubits))
        
        # Validate qubit indices
        if not all(0 <= q < self.n_qubits for q in qubits):
            raise ValueError("Invalid qubit indices")
            
        results = {}
        for qubit in qubits:
            # Calculate probability of measuring |1>
            prob_one = self._calculate_measurement_probability(qubit)
            
            # Perform measurement
            result = 1 if np.random.random() < prob_one else 0
            
            # Update state vector to reflect measurement
            self._collapse_state(qubit, result)
            
            # Store result
            results[qubit] = result
            self.measurements[qubit] = result
            
        return results
    
    def get_state(self) -> np.ndarray:
        """
        Get the current quantum state vector.
        
        Returns:
            Complex numpy array representing quantum state
        """
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset the quantum register to initial state |0>^⊗n."""
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        self.gate_history.clear()
        self.measurements.clear()
    
    def _calculate_measurement_probability(self, qubit: int) -> float:
        """
        Calculate probability of measuring |1> for given qubit.
        
        Args:
            qubit: Index of qubit to measure
            
        Returns:
            Probability of measuring |1>
        """
        # Create projection operator for |1> state
        proj_one = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        for i in range(2**self.n_qubits):
            if (i >> qubit) & 1:
                proj_one[i, i] = 1
                
        # Calculate probability
        return float(np.real(np.dot(np.conj(self.state), np.dot(proj_one, self.state))))
    
    def _collapse_state(self, qubit: int, result: int) -> None:
        """
        Collapse quantum state after measurement.
        
        Args:
            qubit: Index of measured qubit
            result: Measurement result (0 or 1)
        """
        # Create projection operator
        proj = np.zeros((2**self.n_qubits, 2**self.n_qubits))
        for i in range(2**self.n_qubits):
            if ((i >> qubit) & 1) == result:
                proj[i, i] = 1
                
        # Project state and normalize
        self.state = np.dot(proj, self.state)
        norm = np.sqrt(np.sum(np.abs(self.state)**2))
        self.state /= norm
