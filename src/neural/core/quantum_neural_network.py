"""
Advanced Quantum Neural Network Implementation

This module implements a sophisticated quantum-classical hybrid neural network
that combines quantum computing principles with deep learning to create
powerful AI models capable of processing quantum information.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import json
from collections import OrderedDict

from ..utils.quantum_gates import QuantumGate, GateType
from .quantum_register import QuantumRegister

class QuantumActivation(Enum):
    """Quantum activation functions."""
    QUANTUM_SIGMOID = "quantum_sigmoid"
    QUANTUM_TANH = "quantum_tanh"
    QUANTUM_RELU = "quantum_relu"
    PHASE_ACTIVATION = "phase_activation"
    AMPLITUDE_ACTIVATION = "amplitude_activation"

class NetworkArchitecture(Enum):
    """Neural network architectures."""
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    QUANTUM_CONVOLUTION = "quantum_conv"
    HYBRID_ATTENTION = "hybrid_attention"

@dataclass
class QuantumLayerConfig:
    """Configuration for quantum layers."""
    n_qubits: int
    depth: int
    entanglement_pattern: str = "linear"
    measurement_basis: str = "computational"
    noise_model: Optional[str] = None
    optimization_level: int = 1

class QuantumLayer(nn.Module):
    """
    Quantum layer that processes classical input through quantum circuits
    and returns classical output.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 n_qubits: int,
                 depth: int = 3,
                 activation: QuantumActivation = QuantumActivation.QUANTUM_SIGMOID):
        super(QuantumLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_qubits = n_qubits
        self.depth = depth
        self.activation = activation
        
        # Quantum register for processing
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Learnable parameters for quantum gates
        self.rotation_params = nn.Parameter(
            torch.randn(depth, n_qubits, 3) * 0.1  # [depth, qubits, xyz_rotations]
        )
        self.entanglement_params = nn.Parameter(
            torch.randn(depth, n_qubits-1) * 0.1  # [depth, qubit_pairs]
        )
        
        # Classical preprocessing and postprocessing
        self.input_encoder = nn.Linear(input_size, n_qubits)
        self.output_decoder = nn.Linear(n_qubits, output_size)
        
        # Batch normalization for stability
        self.input_norm = nn.BatchNorm1d(n_qubits)
        self.output_norm = nn.BatchNorm1d(output_size)
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum layer."""
        batch_size = x.shape[0]
        
        # Encode classical input to quantum amplitudes
        encoded = self.input_encoder(x)
        encoded = self.input_norm(encoded)
        encoded = torch.tanh(encoded)  # Ensure valid amplitude range
        
        # Process each sample in batch through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            sample = encoded[i]
            quantum_output = self._quantum_forward(sample)
            quantum_outputs.append(quantum_output)
        
        # Stack batch results
        quantum_batch = torch.stack(quantum_outputs)
        
        # Decode quantum output to classical
        output = self.output_decoder(quantum_batch)
        output = self.output_norm(output)
        
        return output
    
    def _quantum_forward(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Process single sample through quantum circuit."""
        # Reset quantum register
        self.quantum_register.reset()
        
        # Encode amplitudes into quantum state
        self._encode_amplitudes(amplitudes.detach().numpy())
        
        # Apply parameterized quantum circuit
        self._apply_quantum_circuit()
        
        # Measure and return probabilities
        measurements = self._measure_quantum_state()
        
        return torch.tensor(measurements, dtype=torch.float32)
    
    def _encode_amplitudes(self, amplitudes: np.ndarray):
        """Encode classical amplitudes into quantum state."""
        # Normalize amplitudes
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm
        
        # Apply rotation gates to encode amplitudes
        for i, amplitude in enumerate(amplitudes[:self.n_qubits]):
            if amplitude != 0:
                angle = 2 * np.arcsin(np.clip(np.abs(amplitude), 0, 1))
                self.quantum_register.apply_gate(
                    QuantumGate(GateType.Ry, {'theta': angle}),
                    [i]
                )
                
                # Apply phase if amplitude is negative
                if amplitude < 0:
                    self.quantum_register.apply_gate(
                        QuantumGate(GateType.P, {'phi': np.pi}),
                        [i]
                    )
    
    def _apply_quantum_circuit(self):
        """Apply parameterized quantum circuit."""
        for layer in range(self.depth):
            # Rotation layer
            for qubit in range(self.n_qubits):
                # Extract learnable parameters
                rx_angle = self.rotation_params[layer, qubit, 0].item()
                ry_angle = self.rotation_params[layer, qubit, 1].item()
                rz_angle = self.rotation_params[layer, qubit, 2].item()
                
                # Apply rotations
                self.quantum_register.apply_gate(
                    QuantumGate(GateType.Rx, {'theta': rx_angle}),
                    [qubit]
                )
                self.quantum_register.apply_gate(
                    QuantumGate(GateType.Ry, {'theta': ry_angle}),
                    [qubit]
                )
                self.quantum_register.apply_gate(
                    QuantumGate(GateType.Rz, {'phi': rz_angle}),
                    [qubit]
                )
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                # Controlled rotation with learnable parameter
                angle = self.entanglement_params[layer, i].item()
                self.quantum_register.apply_gate(
                    QuantumGate(GateType.CRy, {'theta': angle}),
                    [i, i + 1]
                )
    
    def _measure_quantum_state(self) -> np.ndarray:
        """Measure quantum state and return measurement probabilities."""
        measurements = []
        
        for qubit in range(self.n_qubits):
            # Get measurement probability for |1⟩ state
            prob = self.quantum_register.measure_probability(qubit)
            measurements.append(prob)
        
        return np.array(measurements)

class QuantumAttentionLayer(nn.Module):
    """
    Quantum-enhanced attention mechanism that uses quantum superposition
    to process multiple attention patterns simultaneously.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 n_qubits: int = 16,
                 dropout: float = 0.1):
        super(QuantumAttentionLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        # Classical attention components
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Quantum processing for attention weights
        self.quantum_register = QuantumRegister(n_qubits)
        self.quantum_attention = QuantumLayer(
            input_size=num_heads,
            output_size=num_heads,
            n_qubits=min(n_qubits, num_heads * 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through quantum attention layer."""
        batch_size, seq_len, _ = query.shape
        
        # Project to query, key, value
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute classical attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply quantum enhancement to attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Process attention weights through quantum layer
        quantum_enhanced_weights = self._quantum_enhance_attention(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(quantum_enhanced_weights, V)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        output = self.output_proj(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output
    
    def _quantum_enhance_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Enhance attention weights using quantum processing."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        enhanced_weights = []
        
        for b in range(batch_size):
            batch_enhanced = []
            for s in range(seq_len):
                # Extract attention pattern for this position
                pattern = attention_weights[b, :, s, s]  # Self-attention weights
                
                # Process through quantum layer
                enhanced_pattern = self.quantum_attention(pattern.unsqueeze(0))
                enhanced_pattern = enhanced_pattern.squeeze(0)
                
                # Apply enhancement to full attention weights
                enhancement_factor = enhanced_pattern.unsqueeze(-1)
                enhanced_seq = attention_weights[b, :, s, :] * enhancement_factor
                batch_enhanced.append(enhanced_seq)
            
            enhanced_weights.append(torch.stack(batch_enhanced, dim=1))
        
        return torch.stack(enhanced_weights, dim=0)

class QuantumConvolutionalLayer(nn.Module):
    """
    Quantum convolutional layer that applies quantum filters
    to process spatial information.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 n_qubits: int = 8,
                 stride: int = 1,
                 padding: int = 1):
        super(QuantumConvolutionalLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.stride = stride
        self.padding = padding
        
        # Classical convolution for preprocessing
        self.classical_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        
        # Quantum processing for each output channel
        self.quantum_filters = nn.ModuleList([
            QuantumLayer(
                input_size=kernel_size * kernel_size,
                output_size=kernel_size * kernel_size,
                n_qubits=n_qubits
            )
            for _ in range(out_channels)
        ])
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum convolutional layer."""
        # Apply classical convolution
        classical_output = self.classical_conv(x)
        
        # Apply quantum enhancement to each channel
        batch_size, channels, height, width = classical_output.shape
        quantum_enhanced = []
        
        for c in range(channels):
            channel_data = classical_output[:, c, :, :]  # [batch, height, width]
            
            # Process patches through quantum filter
            enhanced_channel = self._apply_quantum_filter(
                channel_data, self.quantum_filters[c]
            )
            quantum_enhanced.append(enhanced_channel)
        
        # Stack channels back together
        output = torch.stack(quantum_enhanced, dim=1)
        output = self.batch_norm(output)
        
        return output
    
    def _apply_quantum_filter(self, 
                             channel_data: torch.Tensor,
                             quantum_filter: QuantumLayer) -> torch.Tensor:
        """Apply quantum filter to channel data."""
        batch_size, height, width = channel_data.shape
        
        # Process overlapping patches
        enhanced_patches = []
        
        for h in range(0, height - self.kernel_size + 1, self.stride):
            row_patches = []
            for w in range(0, width - self.kernel_size + 1, self.stride):
                # Extract patch
                patch = channel_data[:, h:h+self.kernel_size, w:w+self.kernel_size]
                patch_flat = patch.view(batch_size, -1)
                
                # Process through quantum filter
                enhanced_patch = quantum_filter(patch_flat)
                enhanced_patch = enhanced_patch.view(
                    batch_size, self.kernel_size, self.kernel_size
                )
                
                row_patches.append(enhanced_patch)
            
            if row_patches:
                enhanced_patches.append(torch.stack(row_patches, dim=-1))
        
        if enhanced_patches:
            # Combine patches back into full feature map
            output = torch.cat([torch.cat(row, dim=-1) for row in enhanced_patches], dim=-2)
            return output
        else:
            return channel_data

class QuantumNeuralNetwork(nn.Module):
    """
    Complete quantum neural network combining multiple quantum layers
    with classical neural network components.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 n_qubits: int = 16,
                 quantum_layers: int = 2,
                 architecture: NetworkArchitecture = NetworkArchitecture.FEEDFORWARD,
                 activation: QuantumActivation = QuantumActivation.QUANTUM_SIGMOID,
                 dropout: float = 0.1):
        super(QuantumNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_qubits = n_qubits
        self.quantum_layers = quantum_layers
        self.architecture = architecture
        self.activation = activation
        
        # Build network layers
        self.layers = nn.ModuleList()
        self._build_network()
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.output_norm = nn.LayerNorm(output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        self.logger = logging.getLogger(__name__)
        
    def _build_network(self):
        """Build the neural network architecture."""
        layer_sizes = [self.input_size] + self.hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]
            
            # Alternate between quantum and classical layers
            if i < self.quantum_layers:
                # Quantum layer
                quantum_layer = QuantumLayer(
                    input_size=input_dim,
                    output_size=output_dim,
                    n_qubits=min(self.n_qubits, max(input_dim, output_dim)),
                    activation=self.activation
                )
                self.layers.append(quantum_layer)
            else:
                # Classical layer
                classical_layer = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                self.layers.append(classical_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum neural network."""
        # Process through all layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation and dropout
            if i < len(self.layers) - 1:  # Not the last layer
                x = self.dropout(x)
        
        # Output layer
        output = self.output_layer(x)
        output = self.output_norm(output)
        
        return output
    
    def get_quantum_state_info(self) -> Dict[str, Any]:
        """Get information about quantum states in the network."""
        quantum_info = {}
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, QuantumLayer):
                quantum_info[f'layer_{i}'] = {
                    'n_qubits': layer.n_qubits,
                    'depth': layer.depth,
                    'activation': layer.activation.value,
                    'current_state_norm': float(
                        np.linalg.norm(layer.quantum_register.get_state())
                    )
                }
        
        return quantum_info
    
    def compute_quantum_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradients with respect to quantum parameters."""
        quantum_gradients = {}
        
        for name, param in self.named_parameters():
            if 'rotation_params' in name or 'entanglement_params' in name:
                if param.grad is not None:
                    quantum_gradients[name] = param.grad.clone()
        
        return quantum_gradients

class QuantumTransformer(nn.Module):
    """
    Quantum-enhanced transformer architecture combining quantum attention
    with classical transformer components.
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 n_qubits: int = 16,
                 max_seq_length: int = 1024,
                 dropout: float = 0.1):
        super(QuantumTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.n_qubits = n_qubits
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            self._create_transformer_layer(i)
            for i in range(num_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_transformer_layer(self, layer_idx: int) -> nn.Module:
        """Create a transformer layer (quantum or classical)."""
        if layer_idx < self.num_layers // 2:  # First half uses quantum attention
            attention = QuantumAttentionLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                n_qubits=self.n_qubits
            )
        else:  # Second half uses classical attention
            attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.1,
                batch_first=True
            )
        
        # Feed-forward network
        feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(0.1)
        )
        
        return nn.ModuleDict({
            'attention': attention,
            'feed_forward': feed_forward,
            'norm1': nn.LayerNorm(self.embed_dim),
            'norm2': nn.LayerNorm(self.embed_dim)
        })
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through quantum transformer."""
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            # Attention
            if isinstance(layer['attention'], QuantumAttentionLayer):
                attn_output = layer['attention'](x, x, x, attention_mask)
            else:
                attn_output, _ = layer['attention'](x, x, x, key_padding_mask=attention_mask)
            
            x = layer['norm1'](x + attn_output)
            
            # Feed-forward
            ff_output = layer['feed_forward'](x)
            x = layer['norm2'](x + ff_output)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits

# Factory functions and utilities
def create_quantum_neural_network(
    input_size: int,
    output_size: int,
    architecture: str = "feedforward",
    n_qubits: int = 16
) -> QuantumNeuralNetwork:
    """Create a quantum neural network with specified architecture."""
    
    architecture_configs = {
        "feedforward": {
            "hidden_sizes": [128, 64, 32],
            "quantum_layers": 2,
            "architecture": NetworkArchitecture.FEEDFORWARD
        },
        "deep": {
            "hidden_sizes": [512, 256, 128, 64, 32],
            "quantum_layers": 3,
            "architecture": NetworkArchitecture.FEEDFORWARD
        },
        "wide": {
            "hidden_sizes": [256, 256, 256],
            "quantum_layers": 2,
            "architecture": NetworkArchitecture.FEEDFORWARD
        }
    }
    
    config = architecture_configs.get(architecture, architecture_configs["feedforward"])
    
    return QuantumNeuralNetwork(
        input_size=input_size,
        hidden_sizes=config["hidden_sizes"],
        output_size=output_size,
        n_qubits=n_qubits,
        quantum_layers=config["quantum_layers"],
        architecture=config["architecture"]
    )

def create_quantum_transformer(
    vocab_size: int,
    embed_dim: int = 512,
    n_qubits: int = 16
) -> QuantumTransformer:
    """Create a quantum transformer model."""
    return QuantumTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_qubits=n_qubits
    )

class QuantumNeuralNetworkTrainer:
    """Trainer class for quantum neural networks."""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 quantum_lr_factor: float = 0.1):
        self.model = model
        self.learning_rate = learning_rate
        self.quantum_lr_factor = quantum_lr_factor
        
        # Separate optimizers for classical and quantum parameters
        classical_params = []
        quantum_params = []
        
        for name, param in model.named_parameters():
            if 'rotation_params' in name or 'entanglement_params' in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)
        
        self.classical_optimizer = torch.optim.Adam(
            classical_params, lr=learning_rate
        )
        self.quantum_optimizer = torch.optim.Adam(
            quantum_params, lr=learning_rate * quantum_lr_factor
        )
        
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, 
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   loss_fn: nn.Module) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        self.classical_optimizer.zero_grad()
        self.quantum_optimizer.zero_grad()
        
        loss.backward()
        
        # Update parameters
        self.classical_optimizer.step()
        self.quantum_optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def validate(self, 
                 val_loader: torch.utils.data.DataLoader,
                 loss_fn: nn.Module) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
