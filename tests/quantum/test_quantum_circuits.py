import pytest
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from itertools import product

from src.quantum.core.quantum_register import QuantumRegister
from src.quantum.utils.gates import (
    HadamardGate,
    PauliXGate,
    PauliYGate,
    PauliZGate
)
from src.optimization.circuits.optimization_circuits import QuantumOptimizationCircuit

@pytest.mark.quantum
class TestQuantumCircuits:
    """Test quantum circuit validation and properties."""
    
    @pytest.mark.parametrize("test_circuit", [
        {
            'name': 'bell_state',
            'n_qubits': 2,
            'gates': [
                ('h', 0),
                ('cnot', 0, 1)
            ],
            'expected_state': np.array([1, 0, 0, 1]) / np.sqrt(2)
        },
        {
            'name': 'ghz_state',
            'n_qubits': 3,
            'gates': [
                ('h', 0),
                ('cnot', 0, 1),
                ('cnot', 1, 2)
            ],
            'expected_state': np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
        }
    ])
    def test_standard_circuits(self, test_circuit: Dict[str, Any]):
        """Test standard quantum circuit implementations."""
        # Initialize quantum register
        quantum_register = QuantumRegister(test_circuit['n_qubits'])
        
        # Apply gates
        for gate in test_circuit['gates']:
            if gate[0] == 'h':
                quantum_register.apply_gate(HadamardGate(), gate[1])
            elif gate[0] == 'cnot':
                control, target = gate[1], gate[2]
                # Construct CNOT matrix
                dim = 2**test_circuit['n_qubits']
                cnot = np.eye(dim, dtype=complex)
                for i in range(dim):
                    if (i >> control) & 1:
                        target_bit = (i >> target) & 1
                        flipped_state = i ^ (1 << target)
                        cnot[i, i] = 0
                        cnot[i, flipped_state] = 1
                quantum_register.apply_gate(cnot, control)
                
        # Verify final state
        final_state = quantum_register.measure()
        assert np.allclose(final_state, test_circuit['expected_state'])
        
    def test_circuit_unitarity(self):
        """Test unitarity of quantum circuits."""
        n_qubits = 3
        circuit = QuantumOptimizationCircuit(n_qubits, n_layers=2)
        
        # Generate random parameters
        parameters = np.random.randn(circuit.n_parameters)
        
        # Get circuit unitary
        quantum_register = QuantumRegister(n_qubits)
        circuit.apply(quantum_register, parameters)
        
        # Get final state for different input states
        input_states = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # |000⟩
            np.array([0, 1, 0, 0, 0, 0, 0, 0]),  # |001⟩
            np.array([0, 0, 1, 0, 0, 0, 0, 0])   # |010⟩
        ]
        
        output_states = []
        for state in input_states:
            quantum_register = QuantumRegister(n_qubits)
            quantum_register.quantum_states = state
            circuit.apply(quantum_register, parameters)
            output_states.append(quantum_register.measure())
            
        # Verify orthogonality and normalization
        for i, j in product(range(len(output_states)), repeat=2):
            inner_product = np.abs(np.vdot(output_states[i], output_states[j]))
            if i == j:
                assert np.abs(inner_product - 1.0) < 1e-6
            else:
                assert inner_product < 1e-6
                
    def test_circuit_composability(self):
        """Test quantum circuit composition properties."""
        n_qubits = 2
        quantum_register = QuantumRegister(n_qubits)
        
        # Create sequence of gates
        gates = [
            (HadamardGate(), 0),
            (PauliXGate(), 1),
            (PauliYGate(), 0),
            (PauliZGate(), 1)
        ]
        
        # Apply gates in forward order
        forward_register = QuantumRegister(n_qubits)
        for gate, qubit in gates:
            forward_register.apply_gate(gate, qubit)
        forward_state = forward_register.measure()
        
        # Apply gates in reverse order with adjoint
        reverse_register = QuantumRegister(n_qubits)
        for gate, qubit in reversed(gates):
            # Create adjoint gate
            if isinstance(gate, (PauliXGate, PauliYGate, PauliZGate)):
                adjoint_gate = gate  # Self-adjoint
            else:
                adjoint_gate = type(gate)()  # Create new instance
            reverse_register.apply_gate(adjoint_gate, qubit)
            
        reverse_state = reverse_register.measure()
        
        # Verify states are adjoint
        assert np.allclose(forward_state, reverse_state.conj())
        
    def test_circuit_error_propagation(self):
        """Test error propagation in quantum circuits."""
        n_qubits = 4
        quantum_register = QuantumRegister(n_qubits)
        
        # Apply series of gates with error injection
        error_locations = [(1, 0.1), (2, 0.2)]  # (qubit, error_magnitude)
        
        # Perfect evolution
        perfect_register = QuantumRegister(n_qubits)
        gates = [HadamardGate(), PauliXGate(), PauliYGate(), PauliZGate()]
        
        for gate in gates:
            perfect_register.apply_gate(gate, 0)
        perfect_state = perfect_register.measure()
        
        # Evolution with errors
        for gate in gates:
            quantum_register.apply_gate(gate, 0)
            # Inject errors
            for qubit, magnitude in error_locations:
                error_matrix = np.eye(2) + magnitude * np.random.randn(2, 2)
                # Ensure matrix is still roughly unitary
                u, _, vh = np.linalg.svd(error_matrix)
                error_matrix = u @ vh
                quantum_register.apply_gate(error_matrix, qubit)
                
        error_state = quantum_register.measure()
        
        # Calculate fidelity between perfect and error states
        fidelity = np.abs(np.vdot(perfect_state, error_state))**2
        
        # Verify error propagation is bounded
        assert fidelity > 0.5  # Should maintain some fidelity
        
    def test_circuit_optimization(self):
        """Test quantum circuit optimization properties."""
        n_qubits = 3
        circuit = QuantumOptimizationCircuit(n_qubits, n_layers=2)
        
        # Generate random parameters
        parameters = np.random.randn(circuit.n_parameters)
        
        # Get circuit properties
        properties = circuit.get_description(parameters)
        
        # Verify circuit depth
        assert properties['depth'] <= 2 * n_qubits * circuit.config.n_layers
        
        # Verify gate count scaling
        n_gates = properties['n_gates']
        expected_gates = circuit.config.n_layers * (
            n_qubits +  # Single qubit gates
            (n_qubits - 1)  # Two-qubit gates
        )
        assert n_gates <= 2 * expected_gates  # Allow some overhead
        
    def test_circuit_entanglement(self):
        """Test entanglement properties of quantum circuits."""
        n_qubits = 3
        quantum_register = QuantumRegister(n_qubits)
        
        # Create maximally entangled state (GHZ)
        h_gate = HadamardGate()
        quantum_register.apply_gate(h_gate, 0)
        
        # Apply CNOTs
        for i in range(n_qubits - 1):
            # Construct CNOT
            dim = 2**n_qubits
            cnot = np.eye(dim, dtype=complex)
            for j in range(dim):
                if (j >> i) & 1:
                    target_bit = (j >> (i+1)) & 1
                    flipped_state = j ^ (1 << (i+1))
                    cnot[j, j] = 0
                    cnot[j, flipped_state] = 1
            quantum_register.apply_gate(cnot, i)
            
        final_state = quantum_register.measure()
        
        # Calculate entanglement entropy
        entropies = []
        for i in range(n_qubits):
            # Reshape state for partial trace
            state_matrix = final_state.reshape([2] * n_qubits)
            
            # Trace out other qubits
            reduced_matrix = np.trace(state_matrix, axis1=i, axis2=i+1)
            
            # Calculate von Neumann entropy
            eigenvalues = np.linalg.eigvalsh(reduced_matrix)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            entropies.append(entropy)
            
        # Verify maximum entanglement
        assert all(e > 0.9 for e in entropies)  # Should be close to 1
        
    @pytest.mark.parametrize("noise_model", [
        {
            'type': 'depolarizing',
            'probability': 0.01
        },
        {
            'type': 'amplitude_damping',
            'gamma': 0.05
        }
    ])
    def test_circuit_noise_resilience(self, noise_model: Dict[str, Any]):
        """Test circuit behavior under noise."""
        n_qubits = 2
        quantum_register = QuantumRegister(n_qubits)
        
        # Apply gates with noise
        gates = [
            (HadamardGate(), 0),
            (PauliXGate(), 1)
        ]
        
        # Perfect evolution
        perfect_register = QuantumRegister(n_qubits)
        for gate, qubit in gates:
            perfect_register.apply_gate(gate, qubit)
        perfect_state = perfect_register.measure()
        
        # Noisy evolution
        for gate, qubit in gates:
            quantum_register.apply_gate(gate, qubit)
            # Apply noise
            if noise_model['type'] == 'depolarizing':
                p = noise_model['probability']
                noise_matrix = np.sqrt(1-p) * np.eye(2) + \
                             np.sqrt(p/3) * (
                                 PauliXGate().get_matrix() +
                                 PauliYGate().get_matrix() +
                                 PauliZGate().get_matrix()
                             )
            else:  # amplitude damping
                gamma = noise_model['gamma']
                noise_matrix = np.array([
                    [1, 0],
                    [0, np.sqrt(1-gamma)]
                ])
            quantum_register.apply_gate(noise_matrix, qubit)
            
        noisy_state = quantum_register.measure()
        
        # Calculate fidelity
        fidelity = np.abs(np.vdot(perfect_state, noisy_state))**2
        
        # Verify noise impact is bounded
        assert fidelity > 1 - 2*noise_model.get('probability', noise_model.get('gamma'))
        
    def test_circuit_reversibility(self):
        """Test quantum circuit reversibility."""
        n_qubits = 3
        quantum_register = QuantumRegister(n_qubits)
        
        # Create random initial state
        initial_state = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
        initial_state = initial_state / np.linalg.norm(initial_state)
        quantum_register.quantum_states = initial_state
        
        # Apply sequence of gates
        gates = [
            (HadamardGate(), 0),
            (PauliXGate(), 1),
            (PauliYGate(), 2),
            (PauliZGate(), 0)
        ]
        
        # Forward evolution
        for gate, qubit in gates:
            quantum_register.apply_gate(gate, qubit)
            
        # Reverse evolution
        for gate, qubit in reversed(gates):
            # Create adjoint gate
            if isinstance(gate, (PauliXGate, PauliYGate, PauliZGate)):
                adjoint_gate = gate  # Self-adjoint
            else:
                adjoint_gate = type(gate)()  # Create new instance
            quantum_register.apply_gate(adjoint_gate, qubit)
            
        final_state = quantum_register.measure()
        
        # Verify state recovery
        fidelity = np.abs(np.vdot(initial_state, final_state))**2
        assert fidelity > 0.99  # Should recover initial state
        
    def test_circuit_parallelization(self):
        """Test quantum circuit parallelization properties."""
        n_qubits = 4
        circuit = QuantumOptimizationCircuit(n_qubits, n_layers=3)
        
        # Generate random parameters
        parameters = np.random.randn(circuit.n_parameters)
        
        # Get circuit description
        description = circuit.get_description(parameters)
        
        # Analyze gate dependencies
        dependencies = self._analyze_gate_dependencies(description)
        
        # Calculate critical path length
        critical_path = self._calculate_critical_path(dependencies)
        
        # Verify parallelization potential
        max_parallel_gates = max(len(layer['single_qubit_gates'])
                               for layer in description['layers'])
        assert max_parallel_gates > 1  # Should have parallel gates
        assert critical_path < description['depth']  # Should be parallelizable
        
    @staticmethod
    def _analyze_gate_dependencies(description: Dict[str, Any]) -> List[List[int]]:
        """Analyze gate dependencies in circuit."""
        n_gates = description['n_gates']
        dependencies = [[] for _ in range(n_gates)]
        
        gate_index = 0
        for layer in description['layers']:
            # Single-qubit gates
            layer_start = gate_index
            gate_index += len(layer['single_qubit_gates'])
            
            # Two-qubit gates
            for gate in layer.get('entangling_gates', []):
                dependencies[gate_index] = list(range(layer_start, gate_index))
                gate_index += 1
                
        return dependencies
        
    @staticmethod
    def _calculate_critical_path(dependencies: List[List[int]]) -> int:
        """Calculate critical path length in circuit."""
        n_gates = len(dependencies)
        if n_gates == 0:
            return 0
            
        # Dynamic programming to find longest path
        path_lengths = [0] * n_gates
        for i in range(n_gates):
            path_lengths[i] = 1 + max(
                [path_lengths[j] for j in dependencies[i]]
                + [0]
            )
            
        return max(path_lengths)
