import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, List, Optional
import numpy as np
import math

class QuantumLRScheduler(_LRScheduler):
    """
    Quantum-aware learning rate scheduler.
    
    This scheduler adjusts learning rates based on:
    - Quantum state fidelity
    - Entanglement measures
    - Quantum circuit depth
    - Classical metrics
    
    Features:
    - Adaptive quantum-classical balance
    - State-dependent scheduling
    - Entanglement-aware adjustments
    - Circuit complexity adaptation
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 mode: str = 'quantum',
                 warmup_epochs: int = 5,
                 quantum_factor: float = 0.1,
                 min_lr: float = 1e-6,
                 patience: int = 10,
                 cooldown: int = 0,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 verbose: bool = False):
        """
        Initialize quantum scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: Scheduling mode ('quantum', 'hybrid', 'classical')
            warmup_epochs: Number of warmup epochs
            quantum_factor: Quantum adjustment factor
            min_lr: Minimum learning rate
            patience: Epochs to wait before adjustment
            cooldown: Epochs to wait after adjustment
            threshold: Improvement threshold
            threshold_mode: Threshold mode ('rel' or 'abs')
            verbose: Whether to print updates
        """
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.quantum_factor = quantum_factor
        self.min_lr = min_lr
        self.patience = patience
        self.cooldown = cooldown
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.verbose = verbose
        
        # Initialize state
        self.current_epoch = 0
        self.best_value = float('inf')
        self.wait_count = 0
        self.cooldown_counter = 0
        
        # Quantum state tracking
        self.quantum_states: List[np.ndarray] = []
        self.fidelities: List[float] = []
        self.entanglement: List[float] = []
        
        super().__init__(optimizer)
        
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """
        Update learning rates.
        
        Args:
            metrics: Current training metrics
        """
        self.current_epoch += 1
        
        # Apply warmup schedule
        if self.current_epoch <= self.warmup_epochs:
            self._apply_warmup()
            return
            
        # Get quantum adjustments
        if self.mode in ['quantum', 'hybrid']:
            quantum_factor = self._calculate_quantum_adjustment()
        else:
            quantum_factor = 1.0
            
        # Check improvement
        current_value = self._get_current_value(metrics)
        
        if self._is_better(current_value):
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        # Apply learning rate adjustment
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif self.wait_count >= self.patience:
            self._adjust_learning_rate(quantum_factor)
            self.cooldown_counter = self.cooldown
            self.wait_count = 0
            
    def state_dict(self) -> Dict[str, Any]:
        """
        Get scheduler state.
        
        Returns:
            Dict[str, Any]: Scheduler state
        """
        return {
            'current_epoch': self.current_epoch,
            'best_value': self.best_value,
            'wait_count': self.wait_count,
            'cooldown_counter': self.cooldown_counter,
            'quantum_states': self.quantum_states,
            'fidelities': self.fidelities,
            'entanglement': self.entanglement
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load scheduler state.
        
        Args:
            state_dict: Scheduler state
        """
        self.current_epoch = state_dict['current_epoch']
        self.best_value = state_dict['best_value']
        self.wait_count = state_dict['wait_count']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.quantum_states = state_dict['quantum_states']
        self.fidelities = state_dict['fidelities']
        self.entanglement = state_dict['entanglement']
        
    def update_quantum_state(self, quantum_state: np.ndarray):
        """
        Update quantum state history.
        
        Args:
            quantum_state: Current quantum state
        """
        self.quantum_states.append(quantum_state)
        
        if len(self.quantum_states) > 1:
            # Calculate state fidelity
            fidelity = self._calculate_fidelity(
                self.quantum_states[-2],
                quantum_state
            )
            self.fidelities.append(fidelity)
            
            # Calculate entanglement
            entanglement = self._calculate_entanglement(quantum_state)
            self.entanglement.append(entanglement)
            
    def _apply_warmup(self):
        """Apply warmup schedule."""
        # Linear warmup
        factor = self.current_epoch / self.warmup_epochs
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr + factor * (
                param_group['initial_lr'] - self.min_lr
            )
            
    def _calculate_quantum_adjustment(self) -> float:
        """
        Calculate quantum adjustment factor.
        
        Returns:
            float: Adjustment factor
        """
        if not self.fidelities or not self.entanglement:
            return 1.0
            
        # Get recent metrics
        recent_fidelity = np.mean(self.fidelities[-5:])
        recent_entanglement = np.mean(self.entanglement[-5:])
        
        # Calculate adjustment based on quantum properties
        fidelity_factor = 1.0 - recent_fidelity  # Lower fidelity = higher adjustment
        entanglement_factor = recent_entanglement  # Higher entanglement = higher adjustment
        
        adjustment = self.quantum_factor * (
            0.7 * fidelity_factor + 0.3 * entanglement_factor
        )
        
        return 1.0 + adjustment
        
    def _calculate_fidelity(self, state1: np.ndarray,
                           state2: np.ndarray) -> float:
        """
        Calculate quantum state fidelity.
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            
        Returns:
            float: Fidelity between states
        """
        return np.abs(np.vdot(state1, state2))
        
    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """
        Calculate quantum state entanglement.
        
        Args:
            state: Quantum state
            
        Returns:
            float: Entanglement measure
        """
        # Reshape to qubit structure
        n_qubits = int(np.log2(len(state)))
        state_matrix = state.reshape([2] * n_qubits)
        
        # Calculate reduced density matrices
        entropies = []
        for i in range(n_qubits):
            # Trace out other qubits
            reduced_matrix = np.trace(state_matrix, axis1=i, axis2=i+1)
            
            # Calculate von Neumann entropy
            eigenvalues = np.linalg.eigvalsh(reduced_matrix)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            entropies.append(entropy)
            
        return np.mean(entropies)
        
    def _get_current_value(self, metrics: Optional[Dict[str, float]]) -> float:
        """
        Get current metric value.
        
        Args:
            metrics: Training metrics
            
        Returns:
            float: Current value
        """
        if metrics is None:
            return float('inf')
            
        # Use validation loss if available
        if 'val_loss' in metrics:
            return metrics['val_loss']
        return metrics.get('loss', float('inf'))
        
    def _is_better(self, current: float) -> bool:
        """
        Check if current value is better than best.
        
        Args:
            current: Current value
            
        Returns:
            bool: True if value is better
        """
        if self.threshold_mode == 'rel':
            threshold = self.best_value * (1.0 - self.threshold)
        else:
            threshold = self.best_value - self.threshold
            
        return current < threshold
        
    def _adjust_learning_rate(self, quantum_factor: float):
        """
        Adjust learning rate.
        
        Args:
            quantum_factor: Quantum adjustment factor
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * 0.1 * quantum_factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f'Adjusting learning rate from {old_lr:.6f} to {new_lr:.6f}')
                
class CosineQuantumScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with quantum awareness.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0,
                 quantum_factor: float = 0.1,
                 last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            quantum_factor: Quantum adjustment factor
            last_epoch: Last epoch number
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.quantum_factor = quantum_factor
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        Calculate learning rates.
        
        Returns:
            List[float]: Learning rates
        """
        # Calculate cosine schedule
        return [
            self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.last_epoch / self.T_max)
            ) / 2 * (1 + self.quantum_factor)
            for base_lr in self.base_lrs
        ]
        
class QuantumCyclicScheduler(_LRScheduler):
    """
    Cyclic learning rate scheduler with quantum adjustments.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size_up: int,
                 quantum_factor: float = 0.1,
                 mode: str = 'triangular',
                 last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size_up: Steps in up phase
            quantum_factor: Quantum adjustment factor
            mode: LR schedule mode
            last_epoch: Last epoch number
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.quantum_factor = quantum_factor
        self.mode = mode
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        Calculate learning rates.
        
        Returns:
            List[float]: Learning rates
        """
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        
        if self.mode == 'triangular':
            scale_fn = 1.0
        elif self.mode == 'triangular2':
            scale_fn = 1.0 / (2 ** (cycle - 1))
        else:  # exp_range
            scale_fn = 0.5 ** cycle
            
        lr = self.base_lr + (self.max_lr - self.base_lr) * \
             max(0, (1 - x)) * scale_fn * (1 + self.quantum_factor)
             
        return [lr for _ in self.base_lrs]
