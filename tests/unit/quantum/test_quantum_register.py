import pytest
import numpy as np
from typing import List, Dict, Any

from src.quantum.core.quantum_register import QuantumRegister
from src.quantum.utils.gates import (
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate
)

@pytest.mark.quantum
class TestQuantumRegister:
    """Test quantum register functionality."""
    
    def test_initialization(self, quantum_register: QuantumRegister):
        """Test quantum register initialization."""
        # Check initial state
        state = quantum_register.measure()
        assert len(state) == 2**quantum_register.n_qubits
        assert np.allclose(state[0], 1.0)
        assert np.allclose(state[1:], 0.0)
        
    def test_single_qubit_gates(self, quantum_register: QuantumRegister):
        """Test single qubit gate operations."""
        # Test Hadamard gate
        h_gate = HadamardGate()
        quantum_register.apply_gate(h_gate, 0)
        state = quantum_register.measure()
        expected_state = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(state[:2], expected_state)
        
        # Test Pauli-X gate
        quantum_register = QuantumRegister(1)
        x_gate = PauliXGate()
        quantum_register.apply_gate(x_gate, 0)
        state = quantum_register.measure()
        assert np.allclose(state, [0, 1])
        
        # Test Pauli-Y gate
        quantum_register = QuantumRegister(1)
        y_gate = PauliYGate()
        quantum_register.apply_gate(y_gate, 0)
        state = quantum_register.measure()
        assert np.allclose(state, [0, 1j])
        
        # Test Pauli-Z gate
        quantum_register = QuantumRegister(1)
        h_gate = HadamardGate()
        z_gate = PauliZGate()
        quantum_register.apply_gate(h_gate, 0)
        quantum_register.apply_gate(z_gate, 0)
        state = quantum_register.measure()
        expected_state = np.array([1, -1]) / np.sqrt(2)
        assert np.allclose(state, expected_state)
        
    def test_multi_qubit_operations(self, quantum_register: QuantumRegister):
        """Test multi-qubit operations."""
        # Create Bell state
        h_gate = HadamardGate()
        quantum_register.apply_gate(h_gate, 0)
        
        # Apply CNOT
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        quantum_register.apply_gate(cnot_matrix, 0)
        
        state = quantum_register.measure()
        expected_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(state[:4], expected_state)
        
    def test_measurement_statistics(self, quantum_register: QuantumRegister):
        """Test quantum measurement statistics."""
        # Prepare superposition state
        h_gate = HadamardGate()
        quantum_register.apply_gate(h_gate, 0)
        
        # Perform multiple measurements
        n_measurements = 1000
        results = []
        for _ in range(n_measurements):
            quantum_register.apply_gate(h_gate, 0)
            measurement = quantum_register.measure()[0]
            results.append(measurement)
            
        # Check measurement statistics
        zeros = sum(1 for r in results if abs(r - 1/np.sqrt(2)) < 0.1)
        ones = sum(1 for r in results if abs(r + 1/np.sqrt(2)) < 0.1)
        
        # Should be roughly equal (within statistical error)
        assert abs(zeros - n_measurements/2) < n_measurements * 0.1
        assert abs(ones - n_measurements/2) < n_measurements * 0.1
        
    def test_error_handling(self, quantum_register: QuantumRegister):
        """Test error handling in quantum operations."""
        # Test invalid qubit index
        with pytest.raises(ValueError):
            quantum_register.apply_gate(HadamardGate(), quantum_register.n_qubits)
            
        # Test invalid gate matrix size
        invalid_matrix = np.eye(3)
        with pytest.raises(ValueError):
            quantum_register.apply_gate(invalid_matrix, 0)
            
        # Test non-unitary matrix
        non_unitary = np.array([[2, 0], [0, 2]])
        with pytest.raises(ValueError):
            quantum_register.apply_gate(non_unitary, 0)
            
    @pytest.mark.parametrize("n_qubits,gate_sequence", [
        (2, [
            (HadamardGate(), 0),
            (PauliXGate(), 1),
            (PauliYGate(), 0),
            (PauliZGate(), 1)
        ]),
        (3, [
            (HadamardGate(), 0),
            (HadamardGate(), 1),
            (HadamardGate(), 2),
            (PauliXGate(), 0),
            (PauliYGate(), 1),
            (PauliZGate(), 2)
        ])
    ])
    def test_gate_sequences(self, n_qubits: int,
                          gate_sequence: List[tuple]):
        """Test various gate sequences."""
        quantum_register = QuantumRegister(n_qubits)
        
        # Apply gate sequence
        for gate, qubit in gate_sequence:
            quantum_register.apply_gate(gate, qubit)
            
        # Verify state is valid
        state = quantum_register.measure()
        assert len(state) == 2**n_qubits
        assert np.allclose(np.sum(np.abs(state)**2), 1.0)
        
    def test_state_preparation(self, quantum_register: QuantumRegister,
                             random_quantum_state: np.ndarray):
        """Test quantum state preparation."""
        # Prepare specific state
        quantum_register.quantum_states = random_quantum_state
        
        # Verify state
        measured_state = quantum_register.measure()
        assert np.allclose(measured_state, random_quantum_state)
        
    def test_unitary_evolution(self, quantum_register: QuantumRegister,
                             random_unitary_matrix: np.ndarray):
        """Test unitary evolution of quantum state."""
        # Apply unitary transformation
        quantum_register.apply_gate(random_unitary_matrix, 0)
        
        # Verify unitarity is preserved
        state = quantum_register.measure()
        assert np.allclose(np.sum(np.abs(state)**2), 1.0)
        
    @pytest.mark.parametrize("test_circuit", [
        {
            'gates': [
                (HadamardGate(), 0),
                (PauliXGate(), 1),
                (PauliYGate(), 0)
            ],
            'measurements': 100,
            'tolerance': 0.1
        },
        {
            'gates': [
                (HadamardGate(), 0),
                (HadamardGate(), 1),
                (PauliZGate(), 0)
            ],
            'measurements': 100,
            'tolerance': 0.1
        }
    ])
    def test_measurement_reproducibility(self, quantum_register: QuantumRegister,
                                      test_circuit: Dict[str, Any]):
        """Test reproducibility of quantum measurements."""
        measurements1 = []
        measurements2 = []
        
        # First set of measurements
        for _ in range(test_circuit['measurements']):
            # Reset and prepare state
            quantum_register = QuantumRegister(quantum_register.n_qubits)
            for gate, qubit in test_circuit['gates']:
                quantum_register.apply_gate(gate, qubit)
            measurements1.append(quantum_register.measure())
            
        # Second set of measurements
        for _ in range(test_circuit['measurements']):
            # Reset and prepare state
            quantum_register = QuantumRegister(quantum_register.n_qubits)
            for gate, qubit in test_circuit['gates']:
                quantum_register.apply_gate(gate, qubit)
            measurements2.append(quantum_register.measure())
            
        # Compare measurement statistics
        stats1 = np.mean(measurements1, axis=0)
        stats2 = np.mean(measurements2, axis=0)
        assert np.allclose(stats1, stats2, atol=test_circuit['tolerance'])
        
    def test_quantum_register_copy(self, quantum_register: QuantumRegister):
        """Test quantum register copying functionality."""
        # Prepare initial state
        h_gate = HadamardGate()
        quantum_register.apply_gate(h_gate, 0)
        original_state = quantum_register.measure()
        
        # Create copy
        new_register = QuantumRegister(quantum_register.n_qubits)
        new_register.quantum_states = quantum_register.quantum_states.copy()
        
        # Verify states match
        assert np.allclose(new_register.measure(), original_state)
        
        # Modify original should not affect copy
        quantum_register.apply_gate(PauliXGate(), 0)
        assert not np.allclose(
            quantum_register.measure(),
            new_register.measure()
        )
        
    def test_quantum_register_reset(self, quantum_register: QuantumRegister):
        """Test quantum register reset functionality."""
        # Apply some gates
        h_gate = HadamardGate()
        x_gate = PauliXGate()
        quantum_register.apply_gate(h_gate, 0)
        quantum_register.apply_gate(x_gate, 1)
        
        # Reset register
        quantum_register = QuantumRegister(quantum_register.n_qubits)
        
        # Verify reset state
        state = quantum_register.measure()
        assert np.allclose(state[0], 1.0)
        assert np.allclose(state[1:], 0.0)
