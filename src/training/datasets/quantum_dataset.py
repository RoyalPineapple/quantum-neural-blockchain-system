import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pathlib import Path
import h5py
import json

class QuantumDataset(Dataset):
    """
    Dataset class for quantum data handling.
    
    This dataset class supports:
    - Quantum state data
    - Classical-quantum hybrid data
    - On-the-fly quantum state preparation
    - Efficient data loading and caching
    - Data augmentation
    """
    
    def __init__(self,
                 data_path: str,
                 n_qubits: int,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 quantum_transform: Optional[callable] = None,
                 cache_quantum_states: bool = False,
                 max_cache_size: int = 1000):
        """
        Initialize quantum dataset.
        
        Args:
            data_path: Path to dataset
            n_qubits: Number of qubits
            transform: Classical data transform
            target_transform: Target transform
            quantum_transform: Quantum state transform
            cache_quantum_states: Whether to cache quantum states
            max_cache_size: Maximum cache size
        """
        self.data_path = Path(data_path)
        self.n_qubits = n_qubits
        self.transform = transform
        self.target_transform = target_transform
        self.quantum_transform = quantum_transform
        self.cache_quantum_states = cache_quantum_states
        self.max_cache_size = max_cache_size
        
        # Load dataset
        self.data = self._load_data()
        self.quantum_cache = {}
        
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data['quantum_states'])
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (quantum_state, target)
        """
        # Get quantum state
        if self.cache_quantum_states and idx in self.quantum_cache:
            quantum_state = self.quantum_cache[idx]
        else:
            quantum_state = self._prepare_quantum_state(idx)
            if self.cache_quantum_states:
                self._cache_state(idx, quantum_state)
                
        # Get target
        target = self.data['targets'][idx]
        
        # Apply transforms
        if self.transform:
            quantum_state = self.transform(quantum_state)
        if self.target_transform:
            target = self.target_transform(target)
        if self.quantum_transform:
            quantum_state = self.quantum_transform(quantum_state)
            
        return quantum_state, target
        
    def _load_data(self) -> Dict[str, np.ndarray]:
        """
        Load dataset from disk.
        
        Returns:
            Dict[str, np.ndarray]: Dataset contents
        """
        if self.data_path.suffix == '.h5':
            return self._load_hdf5()
        elif self.data_path.suffix == '.npz':
            return self._load_npz()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
    def _load_hdf5(self) -> Dict[str, np.ndarray]:
        """Load HDF5 dataset."""
        with h5py.File(self.data_path, 'r') as f:
            data = {
                'quantum_states': f['quantum_states'][:],
                'classical_data': f['classical_data'][:],
                'targets': f['targets'][:],
                'metadata': json.loads(f.attrs['metadata'])
            }
        return data
        
    def _load_npz(self) -> Dict[str, np.ndarray]:
        """Load NPZ dataset."""
        data = np.load(self.data_path)
        return {
            'quantum_states': data['quantum_states'],
            'classical_data': data['classical_data'],
            'targets': data['targets'],
            'metadata': json.loads(str(data['metadata']))
        }
        
    def _prepare_quantum_state(self, idx: int) -> torch.Tensor:
        """
        Prepare quantum state for given index.
        
        Args:
            idx: Data index
            
        Returns:
            torch.Tensor: Quantum state
        """
        # Get raw state
        raw_state = self.data['quantum_states'][idx]
        
        # Convert to quantum state
        quantum_state = self._encode_quantum_state(raw_state)
        
        # Add classical data if available
        if 'classical_data' in self.data:
            classical_data = self.data['classical_data'][idx]
            quantum_state = self._combine_classical_quantum(
                quantum_state,
                classical_data
            )
            
        return torch.from_numpy(quantum_state)
        
    def _encode_quantum_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encode classical data as quantum state.
        
        Args:
            state: Classical state
            
        Returns:
            np.ndarray: Quantum state
        """
        # Normalize state
        normalized = state / np.linalg.norm(state)
        
        # Ensure correct size
        target_size = 2**self.n_qubits
        if len(normalized) > target_size:
            return normalized[:target_size]
        elif len(normalized) < target_size:
            padding = np.zeros(target_size - len(normalized))
            return np.concatenate([normalized, padding])
        return normalized
        
    def _combine_classical_quantum(self,
                                 quantum_state: np.ndarray,
                                 classical_data: np.ndarray) -> np.ndarray:
        """
        Combine classical and quantum data.
        
        Args:
            quantum_state: Quantum state
            classical_data: Classical data
            
        Returns:
            np.ndarray: Combined state
        """
        # Encode classical data
        classical_quantum = self._encode_quantum_state(classical_data)
        
        # Combine states through tensor product
        combined = np.kron(quantum_state, classical_quantum)
        
        # Normalize
        return combined / np.linalg.norm(combined)
        
    def _cache_state(self, idx: int, state: torch.Tensor):
        """
        Cache quantum state.
        
        Args:
            idx: State index
            state: Quantum state
        """
        # Implement LRU cache
        if len(self.quantum_cache) >= self.max_cache_size:
            # Remove oldest item
            oldest_idx = next(iter(self.quantum_cache))
            del self.quantum_cache[oldest_idx]
            
        self.quantum_cache[idx] = state
        
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dict[str, Any]: Dataset metadata
        """
        return self.data['metadata']
        
    def split(self, split_ratio: float = 0.8) -> Tuple['QuantumDataset', 'QuantumDataset']:
        """
        Split dataset into train and validation sets.
        
        Args:
            split_ratio: Train split ratio
            
        Returns:
            Tuple[QuantumDataset, QuantumDataset]: Train and validation datasets
        """
        # Calculate split indices
        n_train = int(len(self) * split_ratio)
        indices = np.random.permutation(len(self))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create subset datasets
        train_set = QuantumSubset(
            self,
            train_indices,
            transform=self.transform,
            target_transform=self.target_transform,
            quantum_transform=self.quantum_transform
        )
        
        val_set = QuantumSubset(
            self,
            val_indices,
            transform=self.transform,
            target_transform=self.target_transform,
            quantum_transform=self.quantum_transform
        )
        
        return train_set, val_set
        
    def save(self, path: str):
        """
        Save dataset to disk.
        
        Args:
            path: Save path
        """
        path = Path(path)
        if path.suffix == '.h5':
            self._save_hdf5(path)
        elif path.suffix == '.npz':
            self._save_npz(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    def _save_hdf5(self, path: Path):
        """Save dataset in HDF5 format."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('quantum_states', data=self.data['quantum_states'])
            f.create_dataset('classical_data', data=self.data['classical_data'])
            f.create_dataset('targets', data=self.data['targets'])
            f.attrs['metadata'] = json.dumps(self.data['metadata'])
            
    def _save_npz(self, path: Path):
        """Save dataset in NPZ format."""
        np.savez(
            path,
            quantum_states=self.data['quantum_states'],
            classical_data=self.data['classical_data'],
            targets=self.data['targets'],
            metadata=json.dumps(self.data['metadata'])
        )
        
class QuantumSubset(Dataset):
    """Subset of quantum dataset."""
    
    def __init__(self,
                 dataset: QuantumDataset,
                 indices: np.ndarray,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 quantum_transform: Optional[callable] = None):
        """
        Initialize quantum subset.
        
        Args:
            dataset: Parent dataset
            indices: Subset indices
            transform: Data transform
            target_transform: Target transform
            quantum_transform: Quantum transform
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self.quantum_transform = quantum_transform
        
    def __len__(self) -> int:
        """Get subset size."""
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get subset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (quantum_state, target)
        """
        return self.dataset[self.indices[idx]]
