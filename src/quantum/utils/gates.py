import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class QuantumGate:
    """
    Base class for quantum gates with matrix representations.
    """
    name: str
    matrix: np.ndarray
    
    def get_matrix(self) -> np.ndarray:
        """
        Returns the gate's matrix representation.
        """
        return self.matrix

class HadamardGate(QuantumGate):
    """
    Hadamard gate implementation - creates superposition states.
    """
    def __init__(self):
        matrix = (1/np.sqrt(2)) * np.array([[1, 1],
                                          [1, -1]], dtype=complex)
        super().__init__("H", matrix)

class PauliXGate(QuantumGate):
    """
    Pauli-X (NOT) gate implementation.
    """
    def __init__(self):
        matrix = np.array([[0, 1],
                          [1, 0]], dtype=complex)
        super().__init__("X", matrix)

class PauliYGate(QuantumGate):
    """
    Pauli-Y gate implementation.
    """
    def __init__(self):
        matrix = np.array([[0, -1j],
                          [1j, 0]], dtype=complex)
        super().__init__("Y", matrix)

class PauliZGate(QuantumGate):
    """
    Pauli-Z gate implementation.
    """
    def __init__(self):
        matrix = np.array([[1, 0],
                          [0, -1]], dtype=complex)
        super().__init__("Z", matrix)

class PhaseGate(QuantumGate):
    """
    Phase (S) gate implementation.
    """
    def __init__(self):
        matrix = np.array([[1, 0],
                          [0, 1j]], dtype=complex)
        super().__init__("S", matrix)

class ControlledGate(QuantumGate):
    """
    Controlled gate implementation - applies gate only if control qubit is |1⟩.
    """
    def __init__(self, base_gate: QuantumGate):
        """
        Create a controlled version of the given gate.
        
        Args:
            base_gate: The gate to make controlled
        """
        base_matrix = base_gate.get_matrix()
        n = len(base_matrix)
        matrix = np.eye(2*n, dtype=complex)
        matrix[n:, n:] = base_matrix
        super().__init__(f"C-{base_gate.name}", matrix)

class CustomGate(QuantumGate):
    """
    Custom quantum gate implementation with arbitrary unitary matrix.
    """
    def __init__(self, name: str, matrix: np.ndarray):
        """
        Create a custom quantum gate.
        
        Args:
            name: Gate name
            matrix: Unitary matrix representing the gate operation
        """
        if not self._is_unitary(matrix):
            raise ValueError("Gate matrix must be unitary")
        super().__init__(name, matrix)
        
    @staticmethod
    def _is_unitary(matrix: np.ndarray) -> bool:
        """
        Check if a matrix is unitary.
        
        Args:
            matrix: Matrix to check
            
        Returns:
            bool: True if matrix is unitary
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        adjoint = matrix.conj().T
        product = np.dot(matrix, adjoint)
        return np.allclose(product, np.eye(len(matrix)))
