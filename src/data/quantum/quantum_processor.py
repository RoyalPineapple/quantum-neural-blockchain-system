import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import json
from datetime import datetime
import pandas as pd
from scipy import stats

class QuantumDataProcessor:
    """
    Process and transform quantum data.
    
    Features:
    - Quantum state preparation
    - Measurement processing
    - Error correction
    - State transformation
    - Data conversion
    - Quantum feature extraction
    """
    
    def __init__(self,
                 n_qubits: int,
                 error_threshold: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize processor.
        
        Args:
            n_qubits: Number of qubits
            error_threshold: Error threshold for correction
            device: Computation device
        """
        self.n_qubits = n_qubits
        self.error_threshold = error_threshold
        self.device = device
        
    def prepare_quantum_state(self,
                            data: np.ndarray,
                            normalize: bool = True) -> np.ndarray:
        """
        Prepare quantum state from classical data.
        
        Args:
            data: Classical data
            normalize: Whether to normalize state
            
        Returns:
            np.ndarray: Quantum state
        """
        # Flatten and convert to complex
        state = data.flatten().astype(np.complex128)
        
        # Pad or truncate to correct size
        target_size = 2**self.n_qubits
        if len(state) > target_size:
            state = state[:target_size]
        elif len(state) < target_size:
            padding = np.zeros(target_size - len(state), dtype=np.complex128)
            state = np.concatenate([state, padding])
            
        # Normalize if requested
        if normalize:
            state = state / np.linalg.norm(state)
            
        return state
        
    def process_measurement(self,
                          measurements: np.ndarray,
                          shots: int = 1000) -> Dict[str, Any]:
        """
        Process quantum measurement results.
        
        Args:
            measurements: Measurement results
            shots: Number of measurement shots
            
        Returns:
            Dict[str, Any]: Processed results
        """
        # Count measurement outcomes
        unique, counts = np.unique(measurements, return_counts=True)
        probabilities = counts / shots
        
        # Calculate statistics
        mean = np.mean(measurements)
        std = np.std(measurements)
        
        # Calculate quantum state tomography
        density_matrix = self._reconstruct_density_matrix(measurements)
        
        return {
            'counts': dict(zip(unique, counts)),
            'probabilities': dict(zip(unique, probabilities)),
            'statistics': {
                'mean': mean,
                'std': std,
                'variance': std**2
            },
            'density_matrix': density_matrix
        }
        
    def apply_error_correction(self,
                             state: np.ndarray,
                             error_model: str = 'depolarizing') -> np.ndarray:
        """
        Apply quantum error correction.
        
        Args:
            state: Quantum state
            error_model: Error model type
            
        Returns:
            np.ndarray: Corrected state
        """
        # Check if correction needed
        if self._calculate_error_rate(state) < self.error_threshold:
            return state
            
        # Apply error correction based on model
        if error_model == 'depolarizing':
            corrected = self._correct_depolarizing_error(state)
        elif error_model == 'bit_flip':
            corrected = self._correct_bit_flip_error(state)
        elif error_model == 'phase_flip':
            corrected = self._correct_phase_flip_error(state)
        else:
            raise ValueError(f"Unknown error model: {error_model}")
            
        return corrected
        
    def transform_state(self,
                       state: np.ndarray,
                       transformation: str,
                       params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply quantum state transformation.
        
        Args:
            state: Quantum state
            transformation: Transformation type
            params: Transformation parameters
            
        Returns:
            np.ndarray: Transformed state
        """
        if params is None:
            params = {}
            
        if transformation == 'rotate':
            return self._rotate_state(state, **params)
        elif transformation == 'entangle':
            return self._entangle_state(state, **params)
        elif transformation == 'measure':
            return self._measure_state(state, **params)
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
            
    def convert_data_format(self,
                          data: Union[np.ndarray, torch.Tensor],
                          format: str = 'numpy') -> Union[np.ndarray, torch.Tensor]:
        """
        Convert between data formats.
        
        Args:
            data: Input data
            format: Target format
            
        Returns:
            Union[np.ndarray, torch.Tensor]: Converted data
        """
        if format == 'numpy':
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return data
        elif format == 'torch':
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(self.device)
            return data
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def extract_quantum_features(self,
                               state: np.ndarray,
                               feature_type: str = 'basic') -> Dict[str, float]:
        """
        Extract quantum features from state.
        
        Args:
            state: Quantum state
            feature_type: Feature extraction type
            
        Returns:
            Dict[str, float]: Extracted features
        """
        features = {}
        
        if feature_type in ['basic', 'all']:
            # Basic quantum properties
            features.update({
                'norm': np.linalg.norm(state),
                'purity': np.abs(np.vdot(state, state)),
                'phase': np.angle(np.mean(state))
            })
            
        if feature_type in ['entanglement', 'all']:
            # Entanglement properties
            features.update(self._calculate_entanglement_features(state))
            
        if feature_type in ['information', 'all']:
            # Quantum information properties
            features.update(self._calculate_information_features(state))
            
        return features
        
    def save_quantum_data(self,
                         data: Dict[str, Any],
                         path: str,
                         format: str = 'hdf5') -> None:
        """
        Save quantum data to file.
        
        Args:
            data: Quantum data
            path: Save path
            format: File format
        """
        path = Path(path)
        
        if format == 'hdf5':
            with h5py.File(path, 'w') as f:
                # Save quantum states
                f.create_dataset('states', data=data['states'])
                
                # Save measurements
                if 'measurements' in data:
                    f.create_dataset('measurements',
                                   data=data['measurements'])
                                   
                # Save metadata
                f.attrs['metadata'] = json.dumps(data.get('metadata', {}))
                
        elif format == 'npz':
            np.savez(path, **data)
            
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def load_quantum_data(self,
                         path: str,
                         format: str = 'hdf5') -> Dict[str, Any]:
        """
        Load quantum data from file.
        
        Args:
            path: Load path
            format: File format
            
        Returns:
            Dict[str, Any]: Loaded data
        """
        path = Path(path)
        
        if format == 'hdf5':
            with h5py.File(path, 'r') as f:
                data = {
                    'states': f['states'][:],
                    'metadata': json.loads(f.attrs['metadata'])
                }
                
                if 'measurements' in f:
                    data['measurements'] = f['measurements'][:]
                    
        elif format == 'npz':
            data = dict(np.load(path))
            
        else:
            raise ValueError(f"Unknown format: {format}")
            
        return data
        
    def _calculate_error_rate(self, state: np.ndarray) -> float:
        """Calculate quantum error rate."""
        # Check unitarity violation
        unitarity = np.abs(np.vdot(state, state) - 1.0)
        
        # Check state purity
        purity = np.abs(np.vdot(state, state))
        purity_error = np.abs(purity - 1.0)
        
        return max(unitarity, purity_error)
        
    def _correct_depolarizing_error(self, state: np.ndarray) -> np.ndarray:
        """Correct depolarizing channel errors."""
        # Calculate correction factor
        p = self._estimate_depolarizing_probability(state)
        correction = 1.0 / (1.0 - 4.0/3.0 * p)
        
        # Apply correction
        corrected = correction * state
        
        # Renormalize
        return corrected / np.linalg.norm(corrected)
        
    def _correct_bit_flip_error(self, state: np.ndarray) -> np.ndarray:
        """Correct bit flip errors."""
        # Detect bit flips using syndrome measurements
        syndrome = self._measure_syndrome(state)
        
        # Apply correction based on syndrome
        corrected = state.copy()
        for qubit, flip in enumerate(syndrome):
            if flip:
                # Apply X gate to correct bit flip
                corrected = self._apply_x_gate(corrected, qubit)
                
        return corrected
        
    def _correct_phase_flip_error(self, state: np.ndarray) -> np.ndarray:
        """Correct phase flip errors."""
        # Convert to Z basis
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        z_basis = np.kron(hadamard, np.eye(2**(self.n_qubits-1)))
        z_state = z_basis @ state
        
        # Correct in Z basis
        corrected_z = self._correct_bit_flip_error(z_state)
        
        # Convert back to computational basis
        return z_basis.conj().T @ corrected_z
        
    def _rotate_state(self,
                     state: np.ndarray,
                     angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply rotation to quantum state."""
        if angles is None:
            angles = np.random.uniform(0, 2*np.pi, 3)
            
        # Create rotation matrices
        rx = self._rotation_matrix(angles[0], 'x')
        ry = self._rotation_matrix(angles[1], 'y')
        rz = self._rotation_matrix(angles[2], 'z')
        
        # Apply rotations
        rotated = rz @ ry @ rx @ state
        return rotated
        
    def _entangle_state(self,
                       state: np.ndarray,
                       target_qubits: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create entanglement between qubits."""
        if target_qubits is None:
            # Choose random pair of qubits
            target_qubits = tuple(np.random.choice(
                self.n_qubits, 2, replace=False
            ))
            
        # Create CNOT matrix
        cnot = np.eye(2**self.n_qubits)
        control, target = target_qubits
        
        # Update matrix elements for CNOT operation
        for i in range(2**self.n_qubits):
            if (i >> control) & 1:  # If control qubit is 1
                # Flip target qubit
                j = i ^ (1 << target)
                cnot[i, i] = 0
                cnot[i, j] = 1
                
        return cnot @ state
        
    def _measure_state(self,
                      state: np.ndarray,
                      basis: str = 'computational') -> np.ndarray:
        """Perform quantum measurement."""
        if basis == 'computational':
            # Project onto computational basis states
            probabilities = np.abs(state)**2
            outcome = np.random.choice(len(state), p=probabilities)
            
            # Create post-measurement state
            measured = np.zeros_like(state)
            measured[outcome] = 1.0
            
        elif basis == 'bell':
            # Create Bell basis
            bell_basis = self._create_bell_basis()
            
            # Calculate measurement probabilities
            probabilities = np.abs(bell_basis @ state)**2
            outcome = np.random.choice(len(state), p=probabilities)
            
            # Create post-measurement state
            measured = bell_basis[outcome]
            
        else:
            raise ValueError(f"Unknown measurement basis: {basis}")
            
        return measured
        
    def _reconstruct_density_matrix(self,
                                  measurements: np.ndarray) -> np.ndarray:
        """Perform quantum state tomography."""
        n_measurements = len(measurements)
        dim = 2**self.n_qubits
        
        # Initialize density matrix
        rho = np.zeros((dim, dim), dtype=np.complex128)
        
        # Reconstruct from measurements
        for measurement in measurements:
            # Create projection operator
            proj = np.zeros((dim, dim))
            proj[measurement, measurement] = 1
            
            # Update density matrix
            rho += proj
            
        # Normalize
        return rho / n_measurements
        
    def _calculate_entanglement_features(self,
                                       state: np.ndarray) -> Dict[str, float]:
        """Calculate entanglement-related features."""
        # Reshape state for qubit structure
        state_matrix = state.reshape([2] * self.n_qubits)
        
        features = {}
        
        # Calculate entanglement entropy for each qubit
        for i in range(self.n_qubits):
            # Trace out other qubits
            reduced = np.trace(state_matrix, axis1=i, axis2=i+1)
            
            # Calculate von Neumann entropy
            eigenvalues = np.linalg.eigvalsh(reduced)
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
            features[f'entropy_qubit_{i}'] = entropy
            
        # Calculate average entanglement
        features['average_entanglement'] = np.mean(
            [features[f'entropy_qubit_{i}']
             for i in range(self.n_qubits)]
        )
        
        return features
        
    def _calculate_information_features(self,
                                      state: np.ndarray) -> Dict[str, float]:
        """Calculate quantum information features."""
        features = {}
        
        # Calculate quantum Fisher information
        qfi = self._quantum_fisher_information(state)
        features['fisher_information'] = qfi
        
        # Calculate quantum discord
        discord = self._quantum_discord(state)
        features['quantum_discord'] = discord
        
        # Calculate state complexity
        complexity = self._state_complexity(state)
        features['state_complexity'] = complexity
        
        return features
        
    def _quantum_fisher_information(self, state: np.ndarray) -> float:
        """Calculate quantum Fisher information."""
        # Create density matrix
        rho = np.outer(state, state.conj())
        
        # Calculate eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        
        # Calculate QFI
        qfi = 0.0
        for i in range(len(eigenvalues)):
            for j in range(len(eigenvalues)):
                if eigenvalues[i] + eigenvalues[j] > 0:
                    qfi += (eigenvalues[i] - eigenvalues[j])**2 / \
                          (eigenvalues[i] + eigenvalues[j])
                    
        return qfi
        
    def _quantum_discord(self, state: np.ndarray) -> float:
        """Calculate quantum discord."""
        # Create density matrix
        rho = np.outer(state, state.conj())
        
        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(rho)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
        # Calculate mutual information
        mutual_info = self._mutual_information(state)
        
        # Discord is difference between quantum and classical correlations
        return entropy - mutual_info
        
    def _state_complexity(self, state: np.ndarray) -> float:
        """Calculate quantum state complexity."""
        # Use number of significant amplitudes as complexity measure
        amplitudes = np.abs(state)
        significant = np.sum(amplitudes > 1e-10)
        
        # Normalize by maximum possible complexity
        max_complexity = 2**self.n_qubits
        return significant / max_complexity
        
    def _mutual_information(self, state: np.ndarray) -> float:
        """Calculate quantum mutual information."""
        # Create density matrix
        rho = np.outer(state, state.conj())
        
        # Calculate marginal entropies
        s_a = self._von_neumann_entropy(self._partial_trace(rho, [0]))
        s_b = self._von_neumann_entropy(self._partial_trace(rho, [1]))
        
        # Calculate joint entropy
        s_ab = self._von_neumann_entropy(rho)
        
        return s_a + s_b - s_ab
        
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy."""
        eigenvalues = np.linalg.eigvalsh(rho)
        return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        
    def _partial_trace(self, rho: np.ndarray,
                      keep_qubits: List[int]) -> np.ndarray:
        """Calculate partial trace."""
        n_qubits = int(np.log2(len(rho)))
        dims = [2] * n_qubits
        
        # Reshape density matrix
        rho_reshaped = rho.reshape(dims * 2)
        
        # Calculate trace
        traced_out = list(set(range(n_qubits)) - set(keep_qubits))
        for qubit in reversed(traced_out):
            rho_reshaped = np.trace(rho_reshaped, axis1=qubit, axis2=qubit+n_qubits)
            
        return rho_reshaped
        
    def _rotation_matrix(self, angle: float, axis: str) -> np.ndarray:
        """Create rotation matrix."""
        c = np.cos(angle/2)
        s = np.sin(angle/2)
        
        if axis == 'x':
            return np.array([[c, -1j*s], [-1j*s, c]])
        elif axis == 'y':
            return np.array([[c, -s], [s, c]])
        elif axis == 'z':
            return np.array([[np.exp(-1j*angle/2), 0],
                           [0, np.exp(1j*angle/2)]])
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
            
    def _create_bell_basis(self) -> np.ndarray:
        """Create Bell basis states."""
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        phi_minus = np.array([1, 0, 0, -1]) / np.sqrt(2)
        psi_plus = np.array([0, 1, 1, 0]) / np.sqrt(2)
        psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
        
        return np.vstack([phi_plus, phi_minus, psi_plus, psi_minus])
        
    def _measure_syndrome(self, state: np.ndarray) -> np.ndarray:
        """Measure error syndrome."""
        syndrome = np.zeros(self.n_qubits, dtype=bool)
        
        # Measure stabilizer operators
        for i in range(self.n_qubits-1):
            # Create projector for adjacent qubits
            proj = np.zeros((2**self.n_qubits, 2**self.n_qubits))
            for j in range(2**self.n_qubits):
                if (j >> i) & 1 == (j >> (i+1)) & 1:
                    proj[j, j] = 1
                    
            # Calculate expectation value
            expectation = np.real(state.conj() @ proj @ state)
            syndrome[i] = expectation < 0.5
            
        return syndrome
        
    def _apply_x_gate(self, state: np.ndarray, qubit: int) -> np.ndarray:
        """Apply Pauli X gate to specific qubit."""
        # Create X gate matrix
        x_gate = np.array([[0, 1], [1, 0]])
        
        # Create full operation matrix
        operation = np.eye(1)
        for i in range(self.n_qubits):
            operation = np.kron(operation,
                              x_gate if i == qubit else np.eye(2))
                              
        return operation @ state
        
    def _estimate_depolarizing_probability(self, state: np.ndarray) -> float:
        """Estimate depolarizing channel probability."""
        # Calculate state purity
        purity = np.abs(np.vdot(state, state))
        
        # Relate purity to depolarizing probability
        # For depolarizing channel: ρ' = (1-p)ρ + p/d I
        d = 2**self.n_qubits
        p = (d * (1 - purity)) / (d - 1)
        
        return min(max(p, 0), 1)  # Clamp to [0,1]
