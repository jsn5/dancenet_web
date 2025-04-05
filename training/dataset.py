import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DancePoseDataset(Dataset):
    """Dataset for dance pose sequences."""
    
    def __init__(self, data_path, split='train', split_ratio=0.8, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the .npz dataset file.
            split (str): 'train' or 'val' split.
            split_ratio (float): Ratio of training data (0.0 to 1.0).
            transform (callable, optional): Optional transform to apply to samples.
        """
        # Load data
        data = np.load(data_path)
        input_sequences = data['input_sequences']
        target_sequences = data['target_sequences']
        
        # Split data
        num_samples = len(input_sequences)
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * split_ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Select appropriate indices based on split
        if split == 'train':
            self.input_sequences = input_sequences[train_indices]
            self.target_sequences = target_sequences[train_indices]
        else:  # 'val'
            self.input_sequences = input_sequences[val_indices]
            self.target_sequences = target_sequences[val_indices]
            
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (input_sequence, target_sequence)
        """
        input_sequence = self.input_sequences[idx]
        target_sequence = self.target_sequences[idx]
        
        # Convert to torch tensors
        input_sequence = torch.tensor(input_sequence, dtype=torch.float32)
        target_sequence = torch.tensor(target_sequence, dtype=torch.float32)
        
        # Apply transforms if available
        if self.transform:
            input_sequence = self.transform(input_sequence)
            target_sequence = self.transform(target_sequence)
            
        return input_sequence, target_sequence
        
def get_dataloaders(data_path, batch_size=32, split_ratio=0.8, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        data_path (str): Path to the .npz dataset file.
        batch_size (int): Batch size.
        split_ratio (float): Ratio of training data (0.0 to 1.0).
        num_workers (int): Number of workers for data loading.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = DancePoseDataset(data_path, split='train', split_ratio=split_ratio)
    val_dataset = DancePoseDataset(data_path, split='val', split_ratio=split_ratio)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
