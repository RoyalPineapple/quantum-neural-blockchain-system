import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from ...quantum.core.quantum_register import QuantumRegister

class QuantumImageEncoder(nn.Module):
    """
    Quantum circuit for encoding classical image data into quantum states.
    """
    
    def __init__(self, image_size: Tuple[int, int], n_channels: int, n_qubits: int):
        """
        Initialize quantum image encoder.
        
        Args:
            image_size: (height, width) of input images
            n_channels: Number of image channels
            n_qubits: Number of qubits to use for encoding
        """
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_qubits = n_qubits
        
        # Quantum register for encoding
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Learnable encoding parameters
        self.encoding_params = nn.Parameter(
            torch.randn(n_channels, n_qubits, 3)  # 3 rotation angles per qubit
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode classical image data into quantum state.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Quantum state encoding
        """
        batch_size = x.size(0)
        
        # Normalize input
        x = self._normalize_input(x)
        
        # Initialize quantum states
        quantum_states = torch.zeros(
            batch_size, 2**self.n_qubits, dtype=torch.complex64
        )
        
        # Encode each image in batch
        for i in range(batch_size):
            quantum_states[i] = self._encode_single_image(x[i])
            
        return quantum_states
        
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input images.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        # Scale to [0,1]
        x = x / 255.0 if x.max() > 1.0 else x
        
        # Normalize each channel
        for c in range(self.n_channels):
            channel = x[:, c, :, :]
            x[:, c, :, :] = (channel - channel.mean()) / (channel.std() + 1e-8)
            
        return x
        
    def _encode_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode single image into quantum state.
        
        Args:
            image: Single image tensor [channels, height, width]
            
        Returns:
            torch.Tensor: Quantum state encoding
        """
        # Reset quantum register
        self.quantum_register = QuantumRegister(self.n_qubits)
        
        # Encode each channel
        for c in range(self.n_channels):
            channel_data = image[c].flatten()
            self._encode_channel(channel_data, c)
            
        # Get final quantum state
        return torch.from_numpy(self.quantum_register.measure())
        
    def _encode_channel(self, channel_data: torch.Tensor, channel_idx: int) -> None:
        """
        Encode single channel data into quantum state.
        
        Args:
            channel_data: Flattened channel data
            channel_idx: Channel index
        """
        # Get encoding parameters for this channel
        params = self.encoding_params[channel_idx]
        
        # Apply encoding gates for each qubit
        for qubit in range(self.n_qubits):
            # Calculate rotation angles based on image data
            angles = self._calculate_rotation_angles(
                channel_data, qubit, params[qubit]
            )
            
            # Apply rotation gates
            self.quantum_register.apply_gate(
                self._create_rotation_gate(angles),
                qubit
            )
            
    def _calculate_rotation_angles(self, data: torch.Tensor, 
                                 qubit: int, params: torch.Tensor) -> torch.Tensor:
        """
        Calculate rotation angles for quantum encoding.
        
        Args:
            data: Channel data
            qubit: Qubit index
            params: Encoding parameters
            
        Returns:
            torch.Tensor: Rotation angles
        """
        # Extract relevant data for this qubit
        qubit_data = data[qubit::self.n_qubits]
        
        # Calculate weighted average of data values
        data_weight = torch.mean(qubit_data)
        
        # Combine with learnable parameters
        angles = params * data_weight
        
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

class QuantumImageDecoder(nn.Module):
    """
    Quantum circuit for decoding quantum states back to classical image data.
    """
    
    def __init__(self, image_size: Tuple[int, int], n_channels: int, n_qubits: int):
        """
        Initialize quantum image decoder.
        
        Args:
            image_size: (height, width) of output images
            n_channels: Number of image channels
            n_qubits: Number of qubits used in encoding
        """
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_qubits = n_qubits
        
        # Quantum register for decoding
        self.quantum_register = QuantumRegister(n_qubits)
        
        # Learnable decoding parameters
        self.decoding_params = nn.Parameter(
            torch.randn(n_channels, n_qubits, 3)  # 3 rotation angles per qubit
        )
        
        # Classical reconstruction layers
        self.reconstruct_layers = nn.Sequential(
            nn.Linear(2**n_qubits, 512),
            nn.ReLU(),
            nn.Linear(512, n_channels * image_size[0] * image_size[1])
        )
        
    def forward(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Decode quantum state back to classical image.
        
        Args:
            quantum_state: Quantum state tensor [batch_size, 2**n_qubits]
            
        Returns:
            torch.Tensor: Decoded image tensor [batch_size, channels, height, width]
        """
        batch_size = quantum_state.size(0)
        
        # Decode each quantum state
        decoded_states = []
        for i in range(batch_size):
            decoded_state = self._decode_single_state(quantum_state[i])
            decoded_states.append(decoded_state)
            
        # Stack batch
        decoded_batch = torch.stack(decoded_states)
        
        # Reconstruct classical image
        reconstructed = self.reconstruct_layers(decoded_batch)
        
        # Reshape to image dimensions
        return reconstructed.view(
            batch_size, self.n_channels, self.image_size[0], self.image_size[1]
        )
        
    def _decode_single_state(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Decode single quantum state.
        
        Args:
            quantum_state: Quantum state vector
            
        Returns:
            torch.Tensor: Decoded classical features
        """
        # Initialize quantum register with state
        self.quantum_register = QuantumRegister(self.n_qubits)
        self.quantum_register.quantum_states = quantum_state.numpy()
        
        # Apply inverse quantum operations
        decoded_features = []
        for c in range(self.n_channels):
            channel_features = self._decode_channel(c)
            decoded_features.append(channel_features)
            
        # Combine channel features
        return torch.cat(decoded_features)
        
    def _decode_channel(self, channel_idx: int) -> torch.Tensor:
        """
        Decode quantum features for single channel.
        
        Args:
            channel_idx: Channel index
            
        Returns:
            torch.Tensor: Decoded channel features
        """
        # Get decoding parameters for this channel
        params = self.decoding_params[channel_idx]
        
        # Apply inverse encoding operations
        for qubit in range(self.n_qubits):
            # Calculate inverse rotation angles
            angles = -params[qubit]  # Negative angles for inverse rotation
            
            # Apply inverse rotation gates
            self.quantum_register.apply_gate(
                self._create_rotation_gate(angles),
                qubit
            )
            
        # Measure quantum state
        measured_state = self.quantum_register.measure()
        return torch.from_numpy(measured_state)
        
    def _create_rotation_gate(self, angles: torch.Tensor) -> np.ndarray:
        """
        Create composite rotation gate from angles.
        
        Args:
            angles: Rotation angles [Rx, Ry, Rz]
            
        Returns:
            np.ndarray: Gate matrix
        """
        # Create rotation matrices (same as encoder)
        rx = np.array([[np.cos(angles[0]/2), -1j*np.sin(angles[0]/2)],
                      [-1j*np.sin(angles[0]/2), np.cos(angles[0]/2)]])
                      
        ry = np.array([[np.cos(angles[1]/2), -np.sin(angles[1]/2)],
                      [np.sin(angles[1]/2), np.cos(angles[1]/2)]])
                      
        rz = np.array([[np.exp(-1j*angles[2]/2), 0],
                      [0, np.exp(1j*angles[2]/2)]])
                      
        # Combine rotations in reverse order for inverse operation
        gate = np.dot(rx, np.dot(ry, rz))
        return gate
