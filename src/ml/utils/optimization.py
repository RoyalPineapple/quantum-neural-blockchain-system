import torch
import torch.optim as optim
from typing import Iterator, List, Optional
import numpy as np

class QuantumOptimizer:
    """
    Quantum-aware optimization algorithm for hybrid quantum-classical systems.
    Implements gradient descent with quantum state considerations.
    """
    
    def __init__(self, learning_rate: float, quantum_params: Iterator[torch.Tensor]):
        """
        Initialize quantum optimizer.
        
        Args:
            learning_rate: Learning rate for optimization
            quantum_params: Quantum circuit parameters to optimize
        """
        self.learning_rate = learning_rate
        self.quantum_params = list(quantum_params)
        
        # Classical optimizer for quantum parameters
        self.classical_optimizer = optim.Adam(
            self.quantum_params,
            lr=learning_rate
        )
        
        # Quantum parameter history for trajectory analysis
        self.param_history: List[List[torch.Tensor]] = []
        
        # Quantum gradient scaling factors
        self.quantum_scales = torch.ones(len(self.quantum_params))
        
    def quantum_backward(self, loss: torch.Tensor) -> None:
        """
        Perform quantum-aware backward pass.
        
        Args:
            loss: Loss value to backpropagate
        """
        # Store current parameters
        current_params = [p.clone().detach() for p in self.quantum_params]
        self.param_history.append(current_params)
        
        # Calculate quantum gradients
        self._calculate_quantum_gradients(loss)
        
        # Apply quantum gradient scaling
        self._apply_quantum_scaling()
        
        # Classical backward pass
        loss.backward()
        
    def step(self) -> None:
        """Perform optimization step."""
        # Update quantum parameters
        self.classical_optimizer.step()
        
        # Clear gradients
        self.classical_optimizer.zero_grad()
        
        # Update quantum gradient scaling
        self._update_quantum_scaling()
        
    def state_dict(self) -> dict:
        """
        Get optimizer state.
        
        Returns:
            dict: Optimizer state
        """
        return {
            'classical_optimizer': self.classical_optimizer.state_dict(),
            'quantum_scales': self.quantum_scales,
            'param_history': self.param_history
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load optimizer state.
        
        Args:
            state_dict: Optimizer state to load
        """
        self.classical_optimizer.load_state_dict(state_dict['classical_optimizer'])
        self.quantum_scales = state_dict['quantum_scales']
        self.param_history = state_dict['param_history']
        
    def _calculate_quantum_gradients(self, loss: torch.Tensor) -> None:
        """
        Calculate gradients considering quantum effects.
        
        Args:
            loss: Loss value
        """
        # Calculate parameter shift gradients for quantum parameters
        for i, param in enumerate(self.quantum_params):
            if param.grad is None:
                continue
                
            # Calculate quantum gradient using parameter shift rule
            shifted_param = param.clone()
            shift = np.pi/2  # Quantum parameter shift
            
            # Positive shift
            shifted_param.data += shift
            loss_plus = self._evaluate_loss_with_param(param, shifted_param)
            
            # Negative shift
            shifted_param.data -= 2*shift
            loss_minus = self._evaluate_loss_with_param(param, shifted_param)
            
            # Calculate quantum gradient
            quantum_grad = (loss_plus - loss_minus) / (2*shift)
            
            # Combine with classical gradient
            param.grad = param.grad + quantum_grad
            
    def _evaluate_loss_with_param(self, original_param: torch.Tensor, 
                                shifted_param: torch.Tensor) -> torch.Tensor:
        """
        Evaluate loss with shifted parameter.
        
        Args:
            original_param: Original parameter
            shifted_param: Shifted parameter
            
        Returns:
            torch.Tensor: Loss value
        """
        # Store original parameter value
        original_data = original_param.data.clone()
        
        # Replace with shifted parameter
        original_param.data = shifted_param.data
        
        # Forward pass and loss calculation
        # Note: This requires model forward pass which should be implemented
        # in the main training loop
        
        # Restore original parameter
        original_param.data = original_data
        
        # Return dummy loss for now
        return torch.tensor(0.0)
        
    def _apply_quantum_scaling(self) -> None:
        """Apply quantum gradient scaling factors."""
        for i, param in enumerate(self.quantum_params):
            if param.grad is not None:
                param.grad *= self.quantum_scales[i]
                
    def _update_quantum_scaling(self) -> None:
        """Update quantum gradient scaling factors based on parameter history."""
        if len(self.param_history) < 2:
            return
            
        # Calculate parameter changes
        prev_params = self.param_history[-2]
        current_params = self.param_history[-1]
        
        for i, (prev, current) in enumerate(zip(prev_params, current_params)):
            # Calculate parameter change magnitude
            param_change = torch.norm(current - prev)
            
            # Update scaling factor based on parameter change
            if param_change > 1e-6:  # Threshold to avoid numerical issues
                self.quantum_scales[i] *= 0.95  # Reduce scaling if large changes
            else:
                self.quantum_scales[i] *= 1.05  # Increase scaling if small changes
                
            # Clip scaling factors to reasonable range
            self.quantum_scales[i] = torch.clamp(self.quantum_scales[i], 0.1, 10.0)
