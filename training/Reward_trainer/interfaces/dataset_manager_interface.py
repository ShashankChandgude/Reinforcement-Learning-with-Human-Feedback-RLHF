"""
Dataset Manager Interface

Defines the contract for dataset management and data loading in RLHF training.
Supports different dataset types and data processing strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Iterator
import torch
from torch.utils.data import DataLoader, Dataset


class IDatasetManager(ABC):
    """Interface for dataset management."""
    
    @abstractmethod
    def load_dataset(self, config: Dict[str, Any]) -> Dataset:
        """
        Load the dataset based on configuration.
        
        Args:
            config: Dataset configuration dictionary
            
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If dataset configuration is invalid
            FileNotFoundError: If dataset files are not found
            RuntimeError: If dataset loading fails
        """
        pass
    
    @abstractmethod
    def create_dataloader(self, dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
        """
        Create a DataLoader for the dataset.
        
        Args:
            dataset: The dataset to create DataLoader for
            config: DataLoader configuration dictionary
            
        Returns:
            Configured DataLoader
            
        Raises:
            ValueError: If DataLoader configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            dataset: The dataset to get info for
            
        Returns:
            Dictionary containing dataset information
        """
        pass


class IDataProcessor(ABC):
    """Interface for data processing and transformation."""
    
    @abstractmethod
    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of data.
        
        Args:
            batch: List of data items
            
        Returns:
            Processed batch as dictionary of tensors
        """
        pass
    
    @abstractmethod
    def tokenize_texts(self, texts: List[str], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of texts to tokenize
            config: Tokenization configuration
            
        Returns:
            Dictionary containing tokenized inputs
        """
        pass
    
    @abstractmethod
    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data items.
        
        Args:
            batch: List of data items
            
        Returns:
            Collated batch as dictionary of tensors
        """
        pass


class IDataValidator(ABC):
    """Interface for data validation."""
    
    @abstractmethod
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate the dataset structure and content.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Validate a batch of data.
        
        Args:
            batch: Batch to validate
            
        Returns:
            True if batch is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors.
        
        Returns:
            List of validation error messages
        """
        pass


class IDataAugmentation(ABC):
    """Interface for data augmentation strategies."""
    
    @abstractmethod
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to a batch.
        
        Args:
            batch: Original batch
            
        Returns:
            Augmented batch
        """
        pass
    
    @abstractmethod
    def is_augmentation_enabled(self) -> bool:
        """
        Check if augmentation is enabled.
        
        Returns:
            True if augmentation is enabled, False otherwise
        """
        pass
    
    @abstractmethod
    def configure_augmentation(self, config: Dict[str, Any]) -> None:
        """
        Configure augmentation parameters.
        
        Args:
            config: Augmentation configuration dictionary
        """
        pass


class IDataIterator(ABC):
    """Interface for custom data iteration strategies."""
    
    @abstractmethod
    def create_iterator(self, dataloader: DataLoader, config: Dict[str, Any]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Create a custom iterator for the dataloader.
        
        Args:
            dataloader: DataLoader to iterate over
            config: Iterator configuration
            
        Returns:
            Custom iterator
        """
        pass
    
    @abstractmethod
    def get_batch_size(self) -> int:
        """
        Get the current batch size.
        
        Returns:
            Current batch size
        """
        pass
    
    @abstractmethod
    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size.
        
        Args:
            batch_size: New batch size
        """
        pass
