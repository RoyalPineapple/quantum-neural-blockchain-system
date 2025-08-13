import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from .quantum_neural_layer import QuantumNeuralLayer
from .quantum_transformer import QuantumTransformer

class QuantumNeuralNetwork(nn.Module):
    """
    A comprehensive quantum neural network architecture that combines
    quantum computing with classical deep learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_qubits: int = 4,
        n_quantum_layers: int = 2,
        n_classical_layers: int = 3,
        hidden_dim: int = 256,
        transformer_config: Optional[Dict] = None,
        use_transformer: bool = True,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize quantum neural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_qubits: Number of qubits per quantum layer
            n_quantum_layers: Number of quantum layers
            n_classical_layers: Number of classical layers
            hidden_dim: Hidden dimension size
            transformer_config: Optional transformer configuration
            use_transformer: Whether to use quantum transformer
            dropout: Dropout rate
            device: Computation device
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.device = device
        
        # Input preprocessing
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Quantum layers
        self.quantum_layers = nn.ModuleList([
            QuantumNeuralLayer(
                n_qubits=n_qubits,
                n_quantum_layers=n_quantum_layers,
                n_classical_features=hidden_dim,
                device=device
            )
            for _ in range(n_classical_layers)
        ])
        
        # Quantum transformer
        self.use_transformer = use_transformer
        if use_transformer:
            transformer_config = transformer_config or {}
            self.transformer = QuantumTransformer(
                n_qubits=n_qubits,
                d_model=hidden_dim,
                device=device,
                **transformer_config
            )
        
        # Classical processing layers
        self.classical_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(n_classical_layers - 1)
        ])
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_classical_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through quantum neural network.
        
        Args:
            x: Input tensor [batch_size, seq_length, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_length, output_dim]
        """
        # Input preprocessing
        x = self.input_net(x)
        
        # Process through quantum and classical layers
        for i, (quantum_layer, norm) in enumerate(zip(self.quantum_layers, self.layer_norms)):
            # Quantum transformation
            quantum_out = quantum_layer(x)
            x = norm(x + quantum_out)  # Residual connection
            
            # Classical processing
            if i < len(self.classical_layers):
                classical_out = self.classical_layers[i](x)
                x = x + classical_out  # Residual connection
        
        # Apply transformer if enabled
        if self.use_transformer:
            x = self.transformer(x, mask)
        
        # Output processing
        x = self.output_net(x)
        
        return x
    
    def get_quantum_params(self) -> Dict:
        """Get quantum parameters for analysis."""
        params = {
            'n_qubits': self.n_qubits,
            'quantum_layers': []
        }
        
        for i, layer in enumerate(self.quantum_layers):
            params['quantum_layers'].append({
                f'layer_{i}': layer.get_quantum_params()
            })
            
        return params
    
    def quantum_state_analysis(self, x: torch.Tensor) -> List[Dict]:
        """
        Analyze quantum states through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            List of quantum state statistics for each layer
        """
        states = []
        
        # Input preprocessing
        x = self.input_net(x)
        
        # Analyze each quantum layer
        for i, quantum_layer in enumerate(self.quantum_layers):
            # Get quantum state
            quantum_out = quantum_layer(x)
            
            # Calculate statistics
            stats = {
                f'layer_{i}': {
                    'mean': quantum_out.mean().item(),
                    'std': quantum_out.std().item(),
                    'min': quantum_out.min().item(),
                    'max': quantum_out.max().item(),
                    'norm': torch.norm(quantum_out).item()
                }
            }
            
            states.append(stats)
            x = x + quantum_out
            
        return states
    
    def extra_repr(self) -> str:
        """Return string with extra representation info."""
        return (f'input_dim={self.input_dim}, '
                f'output_dim={self.output_dim}, '
                f'n_qubits={self.n_qubits}, '
                f'use_transformer={self.use_transformer}')
