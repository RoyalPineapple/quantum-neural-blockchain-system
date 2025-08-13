import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import h5py
import json
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralDataProcessor:
    """
    Process and transform neural network data.
    
    Features:
    - Data preprocessing
    - Feature engineering
    - Batch processing
    - Data augmentation
    - Dataset management
    - Performance optimization
    """
    
    def __init__(self,
                 input_size: int,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 device: str = 'cuda'):
        """
        Initialize processor.
        
        Args:
            input_size: Input feature size
            batch_size: Batch size for processing
            num_workers: Number of worker processes
            device: Computation device
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # Initialize preprocessing
        self.scaler = StandardScaler()
        self.feature_processors = {}
        
    def preprocess_data(self,
                       data: Union[np.ndarray, torch.Tensor],
                       preprocessing: Optional[List[str]] = None) -> torch.Tensor:
        """
        Preprocess input data.
        
        Args:
            data: Input data
            preprocessing: List of preprocessing steps
            
        Returns:
            torch.Tensor: Preprocessed data
        """
        if preprocessing is None:
            preprocessing = ['standardize']
            
        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        # Apply preprocessing steps
        for step in preprocessing:
            if step == 'standardize':
                data = self.scaler.fit_transform(data)
            elif step == 'normalize':
                data = data / np.linalg.norm(data, axis=1, keepdims=True)
            elif step == 'min_max':
                data = (data - data.min()) / (data.max() - data.min())
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")
                
        return torch.from_numpy(data).float().to(self.device)
        
    def engineer_features(self,
                        data: Union[np.ndarray, torch.Tensor],
                        feature_type: str = 'basic') -> torch.Tensor:
        """
        Perform feature engineering.
        
        Args:
            data: Input data
            feature_type: Type of features to engineer
            
        Returns:
            torch.Tensor: Engineered features
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        if feature_type == 'basic':
            features = self._basic_features(data)
        elif feature_type == 'statistical':
            features = self._statistical_features(data)
        elif feature_type == 'spectral':
            features = self._spectral_features(data)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        return features.to(self.device)
        
    def create_batches(self,
                      data: Union[np.ndarray, torch.Tensor],
                      labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                      shuffle: bool = True) -> DataLoader:
        """
        Create data batches.
        
        Args:
            data: Input data
            labels: Optional labels
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader: Data loader
        """
        # Create dataset
        if labels is not None:
            dataset = self._create_dataset(data, labels)
        else:
            dataset = self._create_dataset(data)
            
        # Create data loader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def augment_data(self,
                    data: Union[np.ndarray, torch.Tensor],
                    augmentation: str = 'basic') -> torch.Tensor:
        """
        Perform data augmentation.
        
        Args:
            data: Input data
            augmentation: Augmentation type
            
        Returns:
            torch.Tensor: Augmented data
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        if augmentation == 'basic':
            augmented = self._basic_augmentation(data)
        elif augmentation == 'noise':
            augmented = self._noise_augmentation(data)
        elif augmentation == 'geometric':
            augmented = self._geometric_augmentation(data)
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")
            
        return augmented.to(self.device)
        
    def split_dataset(self,
                     data: Union[np.ndarray, torch.Tensor],
                     labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     split_ratio: float = 0.8,
                     stratify: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Split dataset into train and validation sets.
        
        Args:
            data: Input data
            labels: Optional labels
            split_ratio: Train split ratio
            stratify: Whether to stratify split
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation loaders
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
            
        # Split data
        if labels is not None:
            stratify_arg = labels if stratify else None
            train_data, val_data, train_labels, val_labels = train_test_split(
                data, labels,
                train_size=split_ratio,
                stratify=stratify_arg
            )
            
            # Create data loaders
            train_loader = self.create_batches(train_data, train_labels)
            val_loader = self.create_batches(val_data, val_labels,
                                           shuffle=False)
            
        else:
            train_data, val_data = train_test_split(
                data,
                train_size=split_ratio
            )
            
            # Create data loaders
            train_loader = self.create_batches(train_data)
            val_loader = self.create_batches(val_data, shuffle=False)
            
        return train_loader, val_loader
        
    def save_processed_data(self,
                          data: Dict[str, Any],
                          path: str,
                          format: str = 'hdf5') -> None:
        """
        Save processed data to file.
        
        Args:
            data: Processed data
            path: Save path
            format: File format
        """
        path = Path(path)
        
        if format == 'hdf5':
            with h5py.File(path, 'w') as f:
                # Save data
                for key, value in data.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        f.create_dataset(key, data=self._to_numpy(value))
                        
                # Save metadata
                f.attrs['metadata'] = json.dumps(data.get('metadata', {}))
                
        elif format == 'npz':
            np.savez(path, **{
                k: self._to_numpy(v)
                for k, v in data.items()
                if isinstance(v, (np.ndarray, torch.Tensor))
            })
            
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def load_processed_data(self,
                          path: str,
                          format: str = 'hdf5') -> Dict[str, Any]:
        """
        Load processed data from file.
        
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
                    key: f[key][:]
                    for key in f.keys()
                }
                data['metadata'] = json.loads(f.attrs['metadata'])
                
        elif format == 'npz':
            data = dict(np.load(path))
            
        else:
            raise ValueError(f"Unknown format: {format}")
            
        return data
        
    def _basic_features(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate basic features."""
        features = []
        
        # Statistical features
        features.append(torch.mean(data, dim=1, keepdim=True))
        features.append(torch.std(data, dim=1, keepdim=True))
        features.append(torch.median(data, dim=1).values.unsqueeze(1))
        
        # Range features
        features.append(torch.max(data, dim=1).values.unsqueeze(1))
        features.append(torch.min(data, dim=1).values.unsqueeze(1))
        
        return torch.cat(features, dim=1)
        
    def _statistical_features(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate statistical features."""
        features = []
        
        # Basic statistics
        features.append(torch.mean(data, dim=1, keepdim=True))
        features.append(torch.std(data, dim=1, keepdim=True))
        features.append(torch.median(data, dim=1).values.unsqueeze(1))
        
        # Higher order statistics
        features.append(torch.pow(data - data.mean(dim=1, keepdim=True), 3).mean(dim=1, keepdim=True))  # Skewness
        features.append(torch.pow(data - data.mean(dim=1, keepdim=True), 4).mean(dim=1, keepdim=True))  # Kurtosis
        
        # Quantile features
        q = torch.tensor([0.25, 0.75])
        features.append(torch.quantile(data, q, dim=1).t())
        
        return torch.cat(features, dim=1)
        
    def _spectral_features(self, data: torch.Tensor) -> torch.Tensor:
        """Calculate spectral features."""
        features = []
        
        # Fourier transform
        fft = torch.fft.fft(data, dim=1)
        
        # Spectral power
        power = torch.abs(fft)
        features.append(torch.mean(power, dim=1, keepdim=True))
        features.append(torch.std(power, dim=1, keepdim=True))
        
        # Peak frequency
        peak_freq = torch.argmax(power, dim=1).unsqueeze(1)
        features.append(peak_freq.float() / data.size(1))
        
        # Spectral entropy
        prob = power / torch.sum(power, dim=1, keepdim=True)
        entropy = -torch.sum(prob * torch.log2(prob + 1e-10), dim=1, keepdim=True)
        features.append(entropy)
        
        return torch.cat(features, dim=1)
        
    def _basic_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Perform basic augmentation."""
        augmented = []
        
        # Original data
        augmented.append(data)
        
        # Add noise
        noise = torch.randn_like(data) * 0.1
        augmented.append(data + noise)
        
        # Scale
        augmented.append(data * 1.1)
        augmented.append(data * 0.9)
        
        return torch.cat(augmented)
        
    def _noise_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Perform noise-based augmentation."""
        augmented = []
        
        # Original data
        augmented.append(data)
        
        # Gaussian noise
        noise = torch.randn_like(data) * 0.1
        augmented.append(data + noise)
        
        # Salt and pepper noise
        mask = torch.rand_like(data) < 0.1
        salt_pepper = data.clone()
        salt_pepper[mask] = torch.randint(2, size=mask.sum().shape) * 2 - 1
        augmented.append(salt_pepper)
        
        # Multiplicative noise
        mult_noise = 1 + torch.randn_like(data) * 0.1
        augmented.append(data * mult_noise)
        
        return torch.cat(augmented)
        
    def _geometric_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Perform geometric augmentation."""
        augmented = []
        
        # Original data
        augmented.append(data)
        
        # Flip
        augmented.append(torch.flip(data, dims=[1]))
        
        # Shift
        pad = torch.nn.ZeroPad2d((1, 1, 0, 0))
        shifted = pad(data)
        augmented.append(shifted[..., :-2])  # Left shift
        augmented.append(shifted[..., 2:])   # Right shift
        
        # Scale
        scale_factors = [0.9, 1.1]
        for scale in scale_factors:
            scaled = torch.nn.functional.interpolate(
                data.unsqueeze(1),
                scale_factor=scale,
                mode='linear'
            ).squeeze(1)
            if scaled.size(1) == data.size(1):
                augmented.append(scaled)
                
        return torch.cat([a for a in augmented if a.size(1) == data.size(1)])
        
    def _create_dataset(self,
                       data: Union[np.ndarray, torch.Tensor],
                       labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dataset:
        """Create PyTorch dataset."""
        # Convert to tensors
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
                
            dataset = torch.utils.data.TensorDataset(data, labels)
        else:
            dataset = torch.utils.data.TensorDataset(data)
            
        return dataset
        
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data
