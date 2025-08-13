import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from ...quantum.core.quantum_register import QuantumRegister

class QuantumTextEncoder(nn.Module):
    """
    Quantum circuit for encoding text embeddings into quantum states.
    """
    
    def __init__(self, embedding_dim: int, n_qubits: int):
        """
        Initialize quantum text encoder.
        
        Args:
            embedding_dim: Dimension of text embeddings
            n_qubits: Number of qubits to use for encoding
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits
        
        # Quantum register for encoding
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Learnable encoding parameters
        self.encoding_params = nn.Parameter(
            torch.randn(embedding_dim, n_qubits, 3)  # 3 rotation angles per qubit
        )
        
        # Dimension reduction if needed
        if embedding_dim > 2**n_qubits:
            self.dim_reduction = nn.Linear(embedding_dim, 2**n_qubits)
        else:
            self.dim_reduction = None
            
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode text embeddings into quantum states.
        
        Args:
            embeddings: Text embeddings [batch_size, seq_length, embedding_dim]
            
        Returns:
            torch.Tensor: Quantum state encodings
        """
        batch_size, seq_length, _ = embeddings.size()
        
        # Dimension reduction if needed
        if self.dim_reduction is not None:
            embeddings = self.dim_reduction(embeddings)
            
        # Initialize quantum states
        quantum_states = torch.zeros(
            batch_size, seq_length, 2**self.n_qubits,
            dtype=torch.complex64
        )
        
        # Encode each embedding
        for i in range(batch_size):
            for j in range(seq_length):
                quantum_states[i,j] = self._encode_single_embedding(
                    embeddings[i,j]
                )
                
        return quantum_states
        
    def _encode_single_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Encode single embedding into quantum state.
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            torch.Tensor: Quantum state encoding
        """
        # Reset quantum register
        self.quantum_register = QuantumRegister(self.n_qubits)
        
        # Apply encoding operations
        for i in range(self.embedding_dim):
            # Calculate rotation angles based on embedding value
            angles = self._calculate_rotation_angles(
                embedding[i],
                self.encoding_params[i]
            )
            
            # Apply rotation gates to each qubit
            for qubit in range(self.n_qubits):
                self.quantum_register.apply_gate(
                    self._create_rotation_gate(angles[qubit]),
                    qubit
                )
                
        # Get final quantum state
        return torch.from_numpy(self.quantum_register.measure())
        
    def _calculate_rotation_angles(self, embedding_value: torch.Tensor,
                                 params: torch.Tensor) -> torch.Tensor:
        """
        Calculate rotation angles for quantum encoding.
        
        Args:
            embedding_value: Single embedding dimension value
            params: Encoding parameters for this dimension
            
        Returns:
            torch.Tensor: Rotation angles for each qubit
        """
        # Scale embedding value to [-π,π]
        scaled_value = torch.tanh(embedding_value) * np.pi
        
        # Combine with learnable parameters
        angles = params * scaled_value
        
        return angles
        
    def _create_rotation_gate(self, angles: torch.Tensor) -> np.ndarray:
        """
        Create composite rotation gate from angles.
        
        Args:
            angles: Rotation angles [Rx, Ry, Rz]
            
        Returns:
            np.ndarray: Gate matrix
        """
        # Create rotation matrices
        rx = np.array([[np.cos(angles[0]/2), -1j*np.sin(angles[0]/2)],
                      [-1j*np.sin(angles[0]/2), np.cos(angles[0]/2)]])
                      
        ry = np.array([[np.cos(angles[1]/2), -np.sin(angles[1]/2)],
                      [np.sin(angles[1]/2), np.cos(angles[1]/2)]])
                      
        rz = np.array([[np.exp(-1j*angles[2]/2), 0],
                      [0, np.exp(1j*angles[2]/2)]])
                      
        # Combine rotations
        gate = np.dot(rz, np.dot(ry, rx))
        return gate
