from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import time
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

@dataclass
class CallbackState:
    """Training callback state."""
    epoch: int = 0
    batch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    best_metric: float = float('inf')
    start_time: float = field(default_factory=time.time)
    epoch_start_time: float = field(default_factory=time.time)
    batch_start_time: float = field(default_factory=time.time)

class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_training_begin(self, state: CallbackState):
        """Called at training start."""
        pass
        
    def on_training_end(self, state: CallbackState):
        """Called at training end."""
        pass
        
    def on_epoch_begin(self, state: CallbackState):
        """Called at epoch start."""
        pass
        
    def on_epoch_end(self, state: CallbackState):
        """Called at epoch end."""
        pass
        
    def on_batch_begin(self, state: CallbackState):
        """Called at batch start."""
        pass
        
    def on_batch_end(self, state: CallbackState):
        """Called at batch end."""
        pass

class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints based on metrics."""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_freq: int = 1):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save if metric improves
            save_freq: Save frequency in epochs
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, state: CallbackState):
        """Save checkpoint at epoch end."""
        current = state.metrics.get(self.monitor)
        if current is None:
            return
            
        if (state.epoch + 1) % self.save_freq == 0:
            if self.save_best_only:
                if self._is_improvement(current):
                    self.best_value = current
                    self._save_checkpoint(state)
            else:
                self._save_checkpoint(state)
                
    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement."""
        if self.mode == 'min':
            return value < self.best_value
        return value > self.best_value
        
    def _save_checkpoint(self, state: CallbackState):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': state.epoch,
            'model_state_dict': state.model.state_dict(),
            'optimizer_state_dict': state.optimizer.state_dict(),
            'metrics': state.metrics
        }
        
        path = self.filepath / f'checkpoint_epoch_{state.epoch}.pt'
        torch.save(checkpoint, path)

class EarlyStopping(TrainingCallback):
    """Stop training when metric stops improving."""
    
    def __init__(self,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 patience: int = 10,
                 min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        
    def on_epoch_end(self, state: CallbackState):
        """Check for early stopping conditions."""
        current = state.metrics.get(self.monitor)
        if current is None:
            return
            
        if self._is_improvement(current):
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = state.epoch
                self.stop_training = True
                
    def _is_improvement(self, value: float) -> bool:
        """Check if value is an improvement."""
        if self.mode == 'min':
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

class TensorBoardLogger(TrainingCallback):
    """Log metrics to TensorBoard."""
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: TensorBoard log directory
        """
        self.log_dir = Path(log_dir)
        self.writer = None
        
    def on_training_begin(self, state: CallbackState):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)
        
    def on_epoch_end(self, state: CallbackState):
        """Log metrics to TensorBoard."""
        for name, value in state.metrics.items():
            self.writer.add_scalar(name, value, state.epoch)
            
    def on_training_end(self, state: CallbackState):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()

class ProgressLogger(TrainingCallback):
    """Log training progress."""
    
    def __init__(self, print_freq: int = 1):
        """
        Initialize progress logger.
        
        Args:
            print_freq: Print frequency in batches
        """
        self.print_freq = print_freq
        
    def on_epoch_begin(self, state: CallbackState):
        """Log epoch start."""
        print(f"\nEpoch {state.epoch + 1}")
        state.epoch_start_time = time.time()
        
    def on_batch_end(self, state: CallbackState):
        """Log batch progress."""
        if (state.batch + 1) % self.print_freq == 0:
            batch_time = time.time() - state.batch_start_time
            print(f"Batch {state.batch + 1}: {batch_time:.3f}s")
            
    def on_epoch_end(self, state: CallbackState):
        """Log epoch metrics."""
        epoch_time = time.time() - state.epoch_start_time
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in state.metrics.items())
        print(f"Epoch {state.epoch + 1} completed in {epoch_time:.3f}s")
        print(f"Metrics: {metrics_str}")

class MetricsHistory(TrainingCallback):
    """Track training metrics history."""
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize metrics history.
        
        Args:
            filepath: Optional path to save history
        """
        self.filepath = Path(filepath) if filepath else None
        self.history: Dict[str, List[float]] = {}
        
    def on_epoch_end(self, state: CallbackState):
        """Update metrics history."""
        for name, value in state.metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)
            
        if self.filepath:
            self._save_history()
            
    def _save_history(self):
        """Save metrics history."""
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)

class LearningRateScheduler(TrainingCallback):
    """Adjust learning rate during training."""
    
    def __init__(self,
                 schedule: Callable[[int], float],
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 min_lr: float = 1e-6):
        """
        Initialize learning rate scheduler.
        
        Args:
            schedule: Learning rate schedule function
            monitor: Metric to monitor
            mode: 'min' or 'max'
            min_lr: Minimum learning rate
        """
        self.schedule = schedule
        self.monitor = monitor
        self.mode = mode
        self.min_lr = min_lr
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def on_epoch_end(self, state: CallbackState):
        """Adjust learning rate."""
        current = state.metrics.get(self.monitor)
        if current is None:
            return
            
        # Calculate new learning rate
        new_lr = max(self.schedule(state.epoch), self.min_lr)
        
        # Update optimizer learning rate
        for param_group in state.optimizer.param_groups:
            param_group['lr'] = new_lr

class QuantumStateMonitor(TrainingCallback):
    """Monitor quantum state properties during training."""
    
    def __init__(self, log_dir: str, monitor_freq: int = 10):
        """
        Initialize quantum state monitor.
        
        Args:
            log_dir: Log directory
            monitor_freq: Monitoring frequency in batches
        """
        self.log_dir = Path(log_dir)
        self.monitor_freq = monitor_freq
        self.writer = None
        
    def on_training_begin(self, state: CallbackState):
        """Initialize TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)
        
    def on_batch_end(self, state: CallbackState):
        """Monitor quantum states."""
        if (state.batch + 1) % self.monitor_freq == 0:
            # Get quantum states from model
            quantum_states = self._get_quantum_states(state.model)
            
            # Calculate quantum properties
            properties = self._analyze_quantum_states(quantum_states)
            
            # Log properties
            for name, value in properties.items():
                self.writer.add_scalar(
                    f"quantum/{name}",
                    value,
                    state.batch
                )
                
    def on_training_end(self, state: CallbackState):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
            
    def _get_quantum_states(self, model: torch.nn.Module) -> torch.Tensor:
        """Extract quantum states from model."""
        quantum_states = []
        for module in model.modules():
            if hasattr(module, 'quantum_state'):
                quantum_states.append(module.quantum_state)
        return torch.stack(quantum_states)
        
    def _analyze_quantum_states(self, states: torch.Tensor) -> Dict[str, float]:
        """Analyze quantum state properties."""
        properties = {}
        
        # Calculate average state norm
        norms = torch.norm(states, dim=-1)
        properties['average_norm'] = torch.mean(norms).item()
        
        # Calculate state purity
        purities = torch.sum(torch.abs(states)**2, dim=-1)
        properties['average_purity'] = torch.mean(purities).item()
        
        # Calculate entanglement
        entanglement = self._calculate_entanglement(states)
        properties['average_entanglement'] = entanglement
        
        return properties
        
    def _calculate_entanglement(self, states: torch.Tensor) -> float:
        """Calculate average entanglement."""
        n_qubits = int(np.log2(states.size(-1)))
        entropies = []
        
        for state in states:
            # Reshape to qubit structure
            state_matrix = state.view([2] * n_qubits)
            
            # Calculate reduced density matrices
            for i in range(n_qubits):
                reduced_matrix = torch.trace(state_matrix, dim1=i, dim2=i+1)
                eigenvalues = torch.linalg.eigvalsh(reduced_matrix)
                entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues + 1e-10))
                entropies.append(entropy.item())
                
        return np.mean(entropies)

class CallbackManager:
    """Manage and execute training callbacks."""
    
    def __init__(self):
        """Initialize callback manager."""
        self.callbacks: List[TrainingCallback] = []
        self.state = CallbackState()
        
    def add_callback(self, callback: TrainingCallback):
        """
        Add callback.
        
        Args:
            callback: Training callback
        """
        self.callbacks.append(callback)
        
    def set_model(self, model: torch.nn.Module):
        """Set model reference."""
        self.state.model = model
        
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set optimizer reference."""
        self.state.optimizer = optimizer
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update current metrics."""
        self.state.metrics = metrics
        
    def on_training_begin(self):
        """Execute training start callbacks."""
        self.state.start_time = time.time()
        for callback in self.callbacks:
            callback.on_training_begin(self.state)
            
    def on_training_end(self):
        """Execute training end callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(self.state)
            
    def on_epoch_begin(self, epoch: int):
        """Execute epoch start callbacks."""
        self.state.epoch = epoch
        self.state.epoch_start_time = time.time()
        for callback in self.callbacks:
            callback.on_epoch_begin(self.state)
            
    def on_epoch_end(self, epoch: int):
        """Execute epoch end callbacks."""
        self.state.epoch = epoch
        for callback in self.callbacks:
            callback.on_epoch_end(self.state)
            
    def on_batch_begin(self, batch: int):
        """Execute batch start callbacks."""
        self.state.batch = batch
        self.state.batch_start_time = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(self.state)
            
    def on_batch_end(self, batch: int):
        """Execute batch end callbacks."""
        self.state.batch = batch
        for callback in self.callbacks:
            callback.on_batch_end(self.state)
            
    def get_total_time(self) -> float:
        """Get total training time."""
        return time.time() - self.state.start_time
