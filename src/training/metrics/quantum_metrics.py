import torch
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field

@dataclass
class MetricState:
    """State for a single metric."""
    values: List[float] = field(default_factory=list)
    running_sum: float = 0.0
    running_count: int = 0
    best_value: Optional[float] = None
    worst_value: Optional[float] = None

class QuantumMetricCollection:
    """
    Collection of quantum-aware metrics for model evaluation.
    
    This class implements various metrics specifically designed for
    quantum-classical hybrid models, including:
    - State fidelity
    - Quantum loss
    - Entanglement metrics
    - Classical accuracy metrics
    """
    
    def __init__(self):
        """Initialize metric collection."""
        self.metrics = {
            'loss': MetricState(),
            'accuracy': MetricState(),
            'fidelity': MetricState(),
            'entanglement': MetricState(),
            'quantum_error': MetricState()
        }
        
        self.custom_metrics = {}
        
    def update(self, outputs: torch.Tensor,
               targets: torch.Tensor) -> Dict[str, float]:
        """
        Update metrics with new batch results.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dict[str, float]: Current metric values
        """
        # Calculate basic metrics
        batch_metrics = {
            'loss': self._calculate_loss(outputs, targets),
            'accuracy': self._calculate_accuracy(outputs, targets),
            'fidelity': self._calculate_fidelity(outputs, targets),
            'entanglement': self._calculate_entanglement(outputs),
            'quantum_error': self._calculate_quantum_error(outputs)
        }
        
        # Update metric states
        for name, value in batch_metrics.items():
            self._update_metric(name, value)
            
        # Calculate custom metrics
        for name, metric_fn in self.custom_metrics.items():
            value = metric_fn(outputs, targets)
            self._update_metric(name, value)
            batch_metrics[name] = value
            
        return batch_metrics
        
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.values = []
            metric.running_sum = 0.0
            metric.running_count = 0
            metric.best_value = None
            metric.worst_value = None
            
    def get_current(self) -> Dict[str, float]:
        """
        Get current metric values.
        
        Returns:
            Dict[str, float]: Current metric values
        """
        return {
            name: self._get_metric_value(metric)
            for name, metric in self.metrics.items()
        }
        
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get metric history.
        
        Returns:
            Dict[str, List[float]]: Metric history
        """
        return {
            name: metric.values
            for name, metric in self.metrics.items()
        }
        
    def get_best(self) -> Dict[str, float]:
        """
        Get best metric values.
        
        Returns:
            Dict[str, float]: Best metric values
        """
        return {
            name: metric.best_value
            for name, metric in self.metrics.items()
            if metric.best_value is not None
        }
        
    def add_custom_metric(self, name: str, metric_fn: callable):
        """
        Add custom metric.
        
        Args:
            name: Metric name
            metric_fn: Metric calculation function
        """
        self.metrics[name] = MetricState()
        self.custom_metrics[name] = metric_fn
        
    def _calculate_loss(self, outputs: torch.Tensor,
                       targets: torch.Tensor) -> float:
        """Calculate quantum-aware loss."""
        # Basic cross-entropy loss
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Add quantum penalty term
        quantum_penalty = self._calculate_quantum_penalty(outputs)
        
        return loss.item() + quantum_penalty
        
    def _calculate_accuracy(self, outputs: torch.Tensor,
                          targets: torch.Tensor) -> float:
        """Calculate classification accuracy."""
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        return correct / total
        
    def _calculate_fidelity(self, outputs: torch.Tensor,
                           targets: torch.Tensor) -> float:
        """Calculate quantum state fidelity."""
        # Convert to quantum states
        output_state = self._normalize_quantum_state(outputs)
        target_state = self._normalize_quantum_state(targets)
        
        # Calculate fidelity
        fidelity = torch.abs(torch.sum(output_state.conj() * target_state))
        return fidelity.item()
        
    def _calculate_entanglement(self, quantum_state: torch.Tensor) -> float:
        """Calculate entanglement metric."""
        # Reshape to qubit structure
        n_qubits = int(np.log2(quantum_state.size(-1)))
        state_matrix = quantum_state.view([2] * n_qubits)
        
        # Calculate reduced density matrices
        entropies = []
        for i in range(n_qubits):
            # Trace out other qubits
            reduced_matrix = torch.trace(state_matrix, dim1=i, dim2=i+1)
            
            # Calculate von Neumann entropy
            eigenvalues = torch.linalg.eigvalsh(reduced_matrix)
            entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-10))
            entropies.append(entropy.item())
            
        # Return average entropy
        return np.mean(entropies)
        
    def _calculate_quantum_error(self, quantum_state: torch.Tensor) -> float:
        """Calculate quantum error metric."""
        # Check unitarity preservation
        norm = torch.norm(quantum_state)
        unitarity_error = torch.abs(norm - 1.0)
        
        # Check state validity
        validity_error = torch.sum(torch.abs(quantum_state.imag()))
        
        return (unitarity_error + validity_error).item()
        
    def _calculate_quantum_penalty(self, quantum_state: torch.Tensor) -> float:
        """Calculate quantum penalty term."""
        # Unitarity penalty
        norm = torch.norm(quantum_state)
        unitarity_penalty = torch.abs(norm - 1.0)
        
        # Entanglement penalty
        entanglement = self._calculate_entanglement(quantum_state)
        entanglement_penalty = torch.tensor(entanglement)
        
        # Combine penalties
        return (unitarity_penalty + 0.1 * entanglement_penalty).item()
        
    def _normalize_quantum_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize quantum state."""
        return state / torch.norm(state)
        
    def _update_metric(self, name: str, value: float):
        """
        Update metric state.
        
        Args:
            name: Metric name
            value: New value
        """
        metric = self.metrics[name]
        
        # Update running statistics
        metric.values.append(value)
        metric.running_sum += value
        metric.running_count += 1
        
        # Update best/worst values
        if metric.best_value is None or value < metric.best_value:
            metric.best_value = value
        if metric.worst_value is None or value > metric.worst_value:
            metric.worst_value = value
            
    def _get_metric_value(self, metric: MetricState) -> float:
        """
        Get current value for metric.
        
        Args:
            metric: Metric state
            
        Returns:
            float: Current metric value
        """
        if metric.running_count == 0:
            return 0.0
        return metric.running_sum / metric.running_count
