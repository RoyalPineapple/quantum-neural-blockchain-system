import pytest
import numpy as np
import torch
from typing import Dict, Any, List

from src.optimization.core.circuit_optimizer import (
    QuantumCircuitOptimizer,
    CircuitOptimizationConfig
)
from src.optimization.circuits.optimization_circuits import QuantumOptimizationCircuit
from src.optimization.utils.parameter_optimization import ParameterOptimizer

@pytest.mark.optimization
class TestCircuitOptimizer:
    """Test quantum circuit optimization functionality."""
    
    def test_initialization(self, circuit_optimizer: QuantumCircuitOptimizer,
                          circuit_optimization_config: CircuitOptimizationConfig):
        """Test optimizer initialization."""
        assert circuit_optimizer.config.n_qubits == circuit_optimization_config.n_qubits
        assert circuit_optimizer.config.n_layers == circuit_optimization_config.n_layers
        assert circuit_optimizer.config.optimization_steps == circuit_optimization_config.optimization_steps
        
        # Check quantum components
        assert hasattr(circuit_optimizer, 'quantum_register')
        assert hasattr(circuit_optimizer, 'optimization_circuit')
        assert hasattr(circuit_optimizer, 'parameter_optimizer')
        
    def test_circuit_optimization(self, circuit_optimizer: QuantumCircuitOptimizer,
                                random_unitary_matrix: np.ndarray):
        """Test circuit optimization process."""
        # Optimize circuit to implement target unitary
        results = circuit_optimizer.optimize_circuit(random_unitary_matrix)
        
        # Check optimization results
        assert 'optimized_parameters' in results
        assert 'final_fidelity' in results
        assert 'optimization_steps' in results
        assert 'optimization_history' in results
        
        # Verify optimization improved fidelity
        initial_fidelity = results['optimization_history'][0]['metric']
        final_fidelity = results['final_fidelity']
        assert final_fidelity > initial_fidelity
        
    def test_ansatz_optimization(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test quantum ansatz optimization."""
        # Define cost function
        def cost_function(state: np.ndarray) -> float:
            return np.sum(np.abs(state - np.array([1, 0, 0, 0])))
            
        # Optimize ansatz
        results = circuit_optimizer.optimize_ansatz(cost_function)
        
        # Check optimization results
        assert 'optimized_parameters' in results
        assert 'final_cost' in results
        assert 'optimization_steps' in results
        
        # Verify cost decreased
        initial_cost = float('inf')
        if results['optimization_history']:
            initial_cost = results['optimization_history'][0]['metric']
        assert results['final_cost'] < initial_cost
        
    @pytest.mark.parametrize("test_case", [
        {
            'n_qubits': 2,
            'target_unitary': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]) / np.sqrt(2)
        },
        {
            'n_qubits': 3,
            'target_unitary': np.kron(
                np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                np.eye(4)
            )
        }
    ])
    def test_different_circuits(self, circuit_optimizer: QuantumCircuitOptimizer,
                              test_case: Dict[str, Any]):
        """Test optimization of different quantum circuits."""
        results = circuit_optimizer.optimize_circuit(test_case['target_unitary'])
        
        # Verify optimization succeeded
        assert results['final_fidelity'] > 0.9
        
    def test_parameter_optimization(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test parameter optimization strategies."""
        # Test gradient-based optimization
        gradient_results = circuit_optimizer.optimize_circuit(
            np.eye(2**circuit_optimizer.config.n_qubits)
        )
        
        # Change optimization strategy
        circuit_optimizer.config.optimization_strategy = 'evolutionary'
        evolutionary_results = circuit_optimizer.optimize_circuit(
            np.eye(2**circuit_optimizer.config.n_qubits)
        )
        
        # Both strategies should improve fidelity
        assert gradient_results['final_fidelity'] > 0.5
        assert evolutionary_results['final_fidelity'] > 0.5
        
    def test_circuit_analysis(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test circuit analysis functionality."""
        # Optimize simple circuit
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        results = circuit_optimizer.optimize_circuit(target_unitary)
        
        # Analyze optimized circuit
        analysis = circuit_optimizer.analyze_circuit(
            results['optimized_parameters']
        )
        
        # Check analysis results
        assert 'depth' in analysis
        assert 'n_gates' in analysis
        assert 'n_parameters' in analysis
        assert 'gate_types' in analysis
        assert 'unitarity' in analysis
        assert 'condition_number' in analysis
        
    def test_optimization_convergence(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test optimization convergence properties."""
        # Simple target unitary
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        
        # Run optimization with convergence monitoring
        results = circuit_optimizer.optimize_circuit(target_unitary)
        history = results['optimization_history']
        
        # Check convergence
        fidelities = [step['metric'] for step in history]
        assert len(fidelities) > 1
        
        # Fidelity should generally increase
        assert fidelities[-1] > fidelities[0]
        
        # Check convergence rate
        convergence_rate = (fidelities[-1] - fidelities[0]) / len(fidelities)
        assert convergence_rate > 0
        
    def test_circuit_complexity(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test handling of circuit complexity."""
        # Get circuit properties
        properties = circuit_optimizer.optimization_circuit.get_description(
            np.random.randn(circuit_optimizer.optimization_circuit.n_parameters)
        )
        
        # Verify complexity metrics
        assert properties['depth'] <= circuit_optimizer.config.max_depth
        assert properties['n_gates'] > 0
        assert len(properties['gate_types']) > 0
        
    def test_error_handling(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test error handling in optimization process."""
        # Test invalid target unitary
        with pytest.raises(ValueError):
            invalid_unitary = np.random.randn(3, 3)  # Wrong size
            circuit_optimizer.optimize_circuit(invalid_unitary)
            
        # Test invalid parameters
        with pytest.raises(ValueError):
            invalid_params = np.random.randn(100000)  # Too many parameters
            circuit_optimizer.analyze_circuit(invalid_params)
            
    def test_optimization_reproducibility(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test reproducibility of optimization results."""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run optimization twice
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        results1 = circuit_optimizer.optimize_circuit(target_unitary)
        
        np.random.seed(42)
        torch.manual_seed(42)
        results2 = circuit_optimizer.optimize_circuit(target_unitary)
        
        # Results should be identical
        assert np.allclose(
            results1['optimized_parameters'],
            results2['optimized_parameters']
        )
        assert np.abs(results1['final_fidelity'] - results2['final_fidelity']) < 1e-6
        
    def test_state_management(self, circuit_optimizer: QuantumCircuitOptimizer,
                            tmp_path):
        """Test optimizer state management."""
        # Initial optimization
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        initial_results = circuit_optimizer.optimize_circuit(target_unitary)
        
        # Save state
        save_path = tmp_path / "optimizer_state.pt"
        circuit_optimizer.save_state(save_path)
        
        # Load state into new optimizer
        loaded_optimizer = QuantumCircuitOptimizer.load_state(save_path)
        
        # Verify loaded state
        loaded_results = loaded_optimizer.optimize_circuit(target_unitary)
        assert np.allclose(
            loaded_results['optimized_parameters'],
            initial_results['optimized_parameters']
        )
        
    @pytest.mark.parametrize("n_qubits,n_layers", [
        (2, 2),
        (3, 3),
        (4, 2)
    ])
    def test_scaling(self, n_qubits: int, n_layers: int):
        """Test optimizer scaling with system size."""
        config = CircuitOptimizationConfig(
            n_qubits=n_qubits,
            n_layers=n_layers,
            optimization_steps=50,
            learning_rate=0.01,
            convergence_threshold=1e-6,
            max_depth=5,
            error_threshold=1e-5,
            optimization_strategy='gradient'
        )
        
        optimizer = QuantumCircuitOptimizer(config)
        
        # Simple optimization task
        target_unitary = np.eye(2**n_qubits)
        results = optimizer.optimize_circuit(target_unitary)
        
        # Verify optimization completed
        assert results['optimization_steps'] > 0
        assert results['final_fidelity'] > 0.5
        
    def test_quantum_resource_estimation(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test quantum resource estimation functionality."""
        # Optimize simple circuit
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        results = circuit_optimizer.optimize_circuit(target_unitary)
        
        # Analyze resource requirements
        circuit_desc = results['optimized_circuit']
        
        # Check resource metrics
        assert 'n_qubits' in circuit_desc
        assert 'depth' in circuit_desc
        assert 'n_gates' in circuit_desc
        
        # Verify resource constraints
        assert circuit_desc['depth'] <= circuit_optimizer.config.max_depth
        assert circuit_desc['n_qubits'] == circuit_optimizer.config.n_qubits
