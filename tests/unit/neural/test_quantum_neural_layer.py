import pytest
import numpy as np
import torch
from typing import Dict, Any

from src.neural.core.quantum_neural_layer import QuantumNeuralLayer, QuantumNeuralConfig

@pytest.mark.neural
class TestQuantumNeuralLayer:
    """Test quantum neural network layer functionality."""
    
    def test_initialization(self, quantum_neural_layer: QuantumNeuralLayer,
                          quantum_neural_config: QuantumNeuralConfig):
        """Test layer initialization."""
        assert quantum_neural_layer.config.n_qubits == quantum_neural_config.n_qubits
        assert quantum_neural_layer.config.n_quantum_layers == quantum_neural_config.n_quantum_layers
        assert quantum_neural_layer.config.n_classical_layers == quantum_neural_config.n_classical_layers
        
        # Check parameter initialization
        assert hasattr(quantum_neural_layer, 'quantum_params')
        assert isinstance(quantum_neural_layer.quantum_params, torch.nn.Parameter)
        expected_shape = (
            quantum_neural_config.n_quantum_layers,
            quantum_neural_config.n_qubits,
            3  # 3 rotation angles per qubit
        )
        assert quantum_neural_layer.quantum_params.shape == expected_shape
        
    def test_forward_pass(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test forward pass through the layer."""
        batch_size = 4
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Create input tensor
        x = torch.randn(batch_size, input_size)
        
        # Forward pass
        output = quantum_neural_layer(x)
        
        # Check output shape
        assert output.shape == (batch_size, input_size)
        
        # Check output values are valid
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_quantum_processing(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test quantum processing component."""
        batch_size = 2
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Create input tensor
        x = torch.randn(batch_size, input_size)
        
        # Get quantum processed features
        quantum_features = quantum_neural_layer._quantum_processing(x)
        
        # Check quantum features
        assert quantum_features.shape == (batch_size, input_size)
        assert quantum_features.dtype == torch.complex64
        
        # Check quantum state properties
        state_norms = torch.sum(torch.abs(quantum_features)**2, dim=1)
        assert torch.allclose(state_norms, torch.ones_like(state_norms))
        
    def test_classical_processing(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test classical processing components."""
        batch_size = 3
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Test preprocessing
        x = torch.randn(batch_size, input_size)
        pre_processed = quantum_neural_layer._classical_preprocessing(x)
        assert pre_processed.shape == (batch_size, input_size)
        
        # Test postprocessing
        quantum_features = torch.randn(batch_size, input_size, dtype=torch.complex64)
        post_processed = quantum_neural_layer._classical_postprocessing(quantum_features)
        assert post_processed.shape == (batch_size, input_size)
        
    def test_backward_pass(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test backward pass and gradient computation."""
        batch_size = 2
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Create input and target
        x = torch.randn(batch_size, input_size, requires_grad=True)
        target = torch.randn(batch_size, input_size)
        
        # Forward pass
        output = quantum_neural_layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check parameter gradients
        assert quantum_neural_layer.quantum_params.grad is not None
        assert not torch.isnan(quantum_neural_layer.quantum_params.grad).any()
        assert not torch.isinf(quantum_neural_layer.quantum_params.grad).any()
        
    @pytest.mark.parametrize("batch_size,n_qubits", [
        (2, 2),
        (4, 3),
        (8, 4)
    ])
    def test_different_sizes(self, batch_size: int, n_qubits: int):
        """Test layer with different batch sizes and qubit numbers."""
        config = QuantumNeuralConfig(
            n_qubits=n_qubits,
            n_quantum_layers=2,
            n_classical_layers=2,
            learning_rate=0.01,
            quantum_circuit_depth=3
        )
        
        layer = QuantumNeuralLayer(config)
        input_size = 2**n_qubits
        x = torch.randn(batch_size, input_size)
        
        output = layer(x)
        assert output.shape == (batch_size, input_size)
        
    def test_quantum_classical_interface(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test quantum-classical interface functionality."""
        batch_size = 2
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Create classical input
        x = torch.randn(batch_size, input_size)
        
        # Process through interface
        quantum_features = quantum_neural_layer._quantum_processing(x)
        classical_output = quantum_neural_layer._classical_postprocessing(quantum_features)
        
        # Check interface properties
        assert quantum_features.dtype == torch.complex64
        assert classical_output.dtype == x.dtype
        assert classical_output.shape == x.shape
        
    def test_layer_reproducibility(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test reproducibility of layer outputs."""
        batch_size = 4
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Create input
        x = torch.randn(batch_size, input_size)
        
        # Multiple forward passes
        output1 = quantum_neural_layer(x)
        output2 = quantum_neural_layer(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
        
    def test_quantum_parameter_updates(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test quantum parameter updates during training."""
        batch_size = 3
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Initial parameters
        initial_params = quantum_neural_layer.quantum_params.clone()
        
        # Training step
        x = torch.randn(batch_size, input_size)
        target = torch.randn(batch_size, input_size)
        
        optimizer = torch.optim.Adam(quantum_neural_layer.parameters(), lr=0.01)
        
        output = quantum_neural_layer(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        assert not torch.allclose(quantum_neural_layer.quantum_params, initial_params)
        
    def test_error_handling(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test error handling in the layer."""
        batch_size = 2
        input_size = 2**quantum_neural_layer.config.n_qubits
        
        # Test invalid input size
        with pytest.raises(ValueError):
            invalid_input = torch.randn(batch_size, input_size + 1)
            quantum_neural_layer(invalid_input)
            
        # Test invalid input dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(batch_size, input_size, 1)
            quantum_neural_layer(invalid_input)
            
    def test_state_dict(self, quantum_neural_layer: QuantumNeuralLayer):
        """Test state dict functionality."""
        # Get state dict
        state_dict = quantum_neural_layer.state_dict()
        
        # Create new layer
        new_layer = QuantumNeuralLayer(quantum_neural_layer.config)
        
        # Load state dict
        new_layer.load_state_dict(state_dict)
        
        # Check parameters match
        for p1, p2 in zip(quantum_neural_layer.parameters(),
                         new_layer.parameters()):
            assert torch.allclose(p1, p2)
            
    @pytest.mark.parametrize("test_case", [
        {
            'input': torch.randn(2, 16),
            'target': torch.randn(2, 16),
            'n_steps': 5,
            'learning_rate': 0.01
        },
        {
            'input': torch.randn(4, 16),
            'target': torch.randn(4, 16),
            'n_steps': 10,
            'learning_rate': 0.001
        }
    ])
    def test_training_convergence(self, quantum_neural_layer: QuantumNeuralLayer,
                                test_case: Dict[str, Any]):
        """Test training convergence."""
        optimizer = torch.optim.Adam(
            quantum_neural_layer.parameters(),
            lr=test_case['learning_rate']
        )
        
        initial_loss = None
        final_loss = None
        
        for step in range(test_case['n_steps']):
            optimizer.zero_grad()
            output = quantum_neural_layer(test_case['input'])
            loss = torch.nn.functional.mse_loss(output, test_case['target'])
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            
            if step == test_case['n_steps'] - 1:
                final_loss = loss.item()
                
        # Loss should decrease
        assert final_loss < initial_loss
