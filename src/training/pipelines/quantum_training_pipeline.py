import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from ..optimizers.quantum_optimizer import QuantumOptimizer
from ..datasets.quantum_dataset import QuantumDataset
from ..metrics.quantum_metrics import QuantumMetricCollection
from ..callbacks.training_callbacks import CallbackManager
from ..schedulers.quantum_scheduler import QuantumLRScheduler

@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    # Model parameters
    model_name: str
    n_qubits: int
    n_layers: int
    
    # Training parameters
    batch_size: int
    n_epochs: int
    learning_rate: float
    
    # Optimization parameters
    optimizer_type: str
    scheduler_type: str
    weight_decay: float
    
    # Quantum parameters
    quantum_device: str
    error_correction: bool
    measurement_strategy: str
    
    # System parameters
    num_workers: int
    device: str
    checkpoint_dir: str
    log_dir: str

class QuantumTrainingPipeline:
    """
    Quantum-enhanced training pipeline for hybrid quantum-classical models.
    
    This pipeline implements advanced training features including:
    - Quantum gradient computation
    - Error mitigation
    - Distributed training support
    - Automatic mixed precision
    - Checkpoint management
    - Performance monitoring
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.metrics = QuantumMetricCollection()
        self.callbacks = CallbackManager()
        
        # Setup training infrastructure
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Initialize state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
    def train(self, 
              train_dataset: QuantumDataset,
              val_dataset: Optional[QuantumDataset] = None,
              test_dataset: Optional[QuantumDataset] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            test_dataset: Optional test dataset
            
        Returns:
            Dict[str, Any]: Training history and metrics
        """
        self.logger.info("Starting training pipeline...")
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, is_training=True)
        val_loader = self._create_dataloader(val_dataset) if val_dataset else None
        test_loader = self._create_dataloader(test_dataset) if test_dataset else None
        
        # Training loop
        try:
            self.callbacks.on_training_begin()
            
            for epoch in range(self.current_epoch, self.config.n_epochs):
                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch)
                
                # Training phase
                train_metrics = self._train_epoch(train_loader)
                self.logger.info(f"Epoch {epoch} training metrics: {train_metrics}")
                
                # Validation phase
                if val_loader:
                    val_metrics = self._validate(val_loader)
                    self.logger.info(f"Epoch {epoch} validation metrics: {val_metrics}")
                    
                    # Check for improvement
                    if val_metrics['loss'] < self.best_metric:
                        self.best_metric = val_metrics['loss']
                        self._save_checkpoint('best.pth')
                        
                # Update learning rate
                self.scheduler.step()
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(f'epoch_{epoch+1}.pth')
                    
                self.callbacks.on_epoch_end(epoch)
                
            # Final evaluation
            if test_loader:
                test_metrics = self._validate(test_loader)
                self.logger.info(f"Final test metrics: {test_metrics}")
                
            self.callbacks.on_training_end()
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
        return self._get_training_summary()
        
    def _train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            self.callbacks.on_batch_begin(batch_idx)
            
            # Move data to device
            inputs, targets = self._prepare_batch(batch)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
                
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Quantum gradient processing
            self._process_quantum_gradients()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            batch_metrics = self.metrics.update(outputs, targets)
            epoch_metrics.append(batch_metrics)
            
            # Update progress
            self.global_step += 1
            self.callbacks.on_batch_end(batch_idx)
            
        return self._aggregate_metrics(epoch_metrics)
        
    def _validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Update metrics
                batch_metrics = self.metrics.update(outputs, targets)
                val_metrics.append(batch_metrics)
                
        return self._aggregate_metrics(val_metrics)
        
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        # Import model dynamically based on config
        model_class = self._import_model_class(self.config.model_name)
        return model_class(self.config)
        
    def _create_optimizer(self) -> QuantumOptimizer:
        """Create quantum-aware optimizer."""
        return QuantumOptimizer(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
    def _create_scheduler(self) -> QuantumLRScheduler:
        """Create learning rate scheduler."""
        return QuantumLRScheduler(
            self.optimizer,
            mode=self.config.scheduler_type
        )
        
    def _create_dataloader(self, 
                          dataset: Optional[QuantumDataset],
                          is_training: bool = False) -> Optional[torch.utils.data.DataLoader]:
        """Create data loader for dataset."""
        if dataset is None:
            return None
            
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data for training."""
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)
        
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with quantum error correction."""
        raw_loss = nn.functional.cross_entropy(outputs, targets)
        
        if self.config.error_correction:
            return self._apply_error_correction(raw_loss)
        return raw_loss
        
    def _process_quantum_gradients(self):
        """Process gradients using quantum computing."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Apply quantum gradient processing
                quantum_grad = self._quantum_gradient_transform(param.grad)
                param.grad = quantum_grad
                
    def _quantum_gradient_transform(self, gradient: torch.Tensor) -> torch.Tensor:
        """Transform gradients using quantum operations."""
        # Convert to quantum state
        quantum_grad = self._encode_quantum_state(gradient)
        
        # Apply quantum operations
        quantum_grad = self._apply_quantum_operations(quantum_grad)
        
        # Convert back to classical gradient
        return self._decode_quantum_state(quantum_grad)
        
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__
        }
        
        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(
            Path(self.config.log_dir) / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across batches."""
        aggregated = {}
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = np.mean(values)
        return aggregated
        
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'epochs_completed': self.current_epoch + 1,
            'global_steps': self.global_step,
            'best_metric': self.best_metric,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'training_time': self.callbacks.get_total_time(),
            'metrics_history': self.metrics.get_history()
        }
        
    @staticmethod
    def _import_model_class(model_name: str) -> type:
        """Dynamically import model class."""
        components = model_name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod
        
    def _encode_quantum_state(self, tensor: torch.Tensor) -> np.ndarray:
        """Encode classical tensor as quantum state."""
        # Normalize tensor
        normalized = tensor / torch.norm(tensor)
        
        # Convert to quantum state
        quantum_state = normalized.cpu().numpy()
        return quantum_state
        
    def _decode_quantum_state(self, quantum_state: np.ndarray) -> torch.Tensor:
        """Decode quantum state to classical tensor."""
        return torch.from_numpy(quantum_state).to(self.device)
        
    def _apply_quantum_operations(self, quantum_state: np.ndarray) -> np.ndarray:
        """Apply quantum operations to state."""
        # Initialize quantum register
        qreg = QuantumRegister(self.config.n_qubits)
        qreg.quantum_states = quantum_state
        
        # Apply quantum operations based on configuration
        if self.config.measurement_strategy == 'projective':
            return self._apply_projective_measurement(qreg)
        elif self.config.measurement_strategy == 'weak':
            return self._apply_weak_measurement(qreg)
        else:
            return qreg.measure()
            
    def _apply_error_correction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply quantum error correction to loss."""
        # Convert to quantum state
        quantum_loss = self._encode_quantum_state(loss.reshape(-1))
        
        # Apply error correction
        qreg = QuantumRegister(self.config.n_qubits)
        qreg.quantum_states = quantum_loss
        
        # Perform error correction
        corrected_state = qreg.error_correction.correct_state(qreg.quantum_states)
        
        # Convert back to classical loss
        corrected_loss = self._decode_quantum_state(corrected_state)
        return corrected_loss.reshape(loss.shape)
        
    def _apply_projective_measurement(self, qreg: QuantumRegister) -> np.ndarray:
        """Apply projective measurement."""
        return qreg.measure()
        
    def _apply_weak_measurement(self, qreg: QuantumRegister) -> np.ndarray:
        """Apply weak measurement."""
        # Implement weak measurement protocol
        # This is a simplified version
        state = qreg.quantum_states
        measurement_strength = 0.1
        
        # Apply weak measurement
        measured_state = state * (1 - measurement_strength)
        return measured_state
