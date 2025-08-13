import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
from ...quantum.core.quantum_register import QuantumRegister
from ...quantum.utils.gates import QuantumGate, GateType

class QuantumMetrics:
    """
    Quantum-enhanced metrics calculation and tracking system.
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize quantum metrics."""
        self.n_qubits = n_qubits
        self.device = device
        
        # Initialize quantum register
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Metrics history
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'quantum_fidelity': [],
            'entanglement': [],
            'gradient_norm': []
        }
    
    def calculate(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        quantum_state: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Classical metrics
        metrics.update(self._calculate_classical_metrics(outputs, targets))
        
        # Quantum metrics
        if quantum_state is not None:
            metrics.update(self._calculate_quantum_metrics(quantum_state))
        
        # Gradient metrics
        if gradients is not None:
            metrics.update(self._calculate_gradient_metrics(gradients))
        
        # Update history
        self._update_history(metrics)
        
        return metrics
    
    def _calculate_classical_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate classical performance metrics."""
        # Loss
        loss = nn.functional.cross_entropy(outputs, targets)
        
        # Accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def _calculate_quantum_metrics(
        self,
        quantum_state: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate quantum-specific metrics."""
        # Quantum fidelity
        fidelity = torch.abs(torch.vdot(quantum_state, quantum_state))
        
        # Entanglement entropy
        density_matrix = torch.outer(quantum_state, torch.conj(quantum_state))
        eigenvalues = torch.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues))
        
        return {
            'quantum_fidelity': fidelity.item(),
            'entanglement': entropy.item()
        }
    
    def _calculate_gradient_metrics(
        self,
        gradients: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate gradient-based metrics."""
        # Gradient norm
        grad_norm = torch.norm(gradients)
        
        return {
            'gradient_norm': grad_norm.item()
        }
    
    def _update_history(
        self,
        metrics: Dict[str, float]
    ) -> None:
        """Update metrics history."""
        for metric, value in metrics.items():
            if metric in self.history:
                self.history[metric].append(value)
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get metrics history."""
        return self.history
    
    def reset(self) -> None:
        """Reset metrics history."""
        for metric in self.history:
            self.history[metric] = []

class QuantumLoss(nn.Module):
    """Quantum-enhanced loss function."""
    
    def __init__(
        self,
        n_qubits: int,
        alpha: float = 0.5,
        device: str = "cuda"
    ):
        """Initialize quantum loss."""
        super().__init__()
        
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.device = device
        
        # Quantum register
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Loss network
        self.loss_net = nn.Sequential(
            nn.Linear(n_qubits * 2, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, 1)
        ).to(device)
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        quantum_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate hybrid quantum-classical loss."""
        # Classical loss component
        classical_loss = nn.functional.cross_entropy(outputs, targets)
        
        # Quantum loss component
        if quantum_state is not None:
            quantum_loss = self._quantum_loss(quantum_state)
        else:
            quantum_loss = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = (
            (1 - self.alpha) * classical_loss +
            self.alpha * quantum_loss
        )
        
        return total_loss
    
    def _quantum_loss(
        self,
        quantum_state: torch.Tensor
    ) -> torch.Tensor:
        """Calculate quantum loss component."""
        # Reset quantum register
        self.quantum_register.reset()
        
        # Apply quantum operations
        for i in range(self.n_qubits):
            angle = quantum_state[i] * torch.pi
            self.quantum_register.apply_gate(
                QuantumGate(GateType.Ry, {'theta': angle.item()}),
                [i]
            )
        
        # Get quantum state
        final_state = self.quantum_register.get_state()
        
        # Calculate loss
        state_tensor = torch.cat([
            torch.tensor(final_state.real, device=self.device),
            torch.tensor(final_state.imag, device=self.device)
        ])
        
        return self.loss_net(state_tensor)

class QuantumAccuracy(nn.Module):
    """Quantum-enhanced accuracy metric."""
    
    def __init__(
        self,
        n_qubits: int,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        """Initialize quantum accuracy."""
        super().__init__()
        
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.device = device
        
        # Quantum register
        self.quantum_register = QuantumRegister(n_qubits)
    
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        quantum_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate quantum-enhanced accuracy."""
        # Classical accuracy
        predictions = torch.argmax(outputs, dim=1)
        classical_acc = (predictions == targets).float().mean()
        
        # Quantum accuracy component
        if quantum_state is not None:
            quantum_acc = self._quantum_accuracy(quantum_state)
            return (classical_acc + quantum_acc) / 2
        
        return classical_acc
    
    def _quantum_accuracy(
        self,
        quantum_state: torch.Tensor
    ) -> torch.Tensor:
        """Calculate quantum accuracy component."""
        # Reset quantum register
        self.quantum_register.reset()
        
        # Apply quantum operations
        for i in range(self.n_qubits):
            angle = quantum_state[i] * torch.pi
            self.quantum_register.apply_gate(
                QuantumGate(GateType.Ry, {'theta': angle.item()}),
                [i]
            )
        
        # Measure qubits
        measurements = self.quantum_register.measure()
        
        # Calculate accuracy based on measurements
        correct = sum(1 for v in measurements.values() if v > self.threshold)
        return torch.tensor(correct / self.n_qubits, device=self.device)
