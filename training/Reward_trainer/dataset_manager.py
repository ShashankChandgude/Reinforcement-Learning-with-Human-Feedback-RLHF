"""
Dataset Manager Implementations

Concrete implementations of dataset management and data loading for RLHF training.
Supports different dataset types and data processing strategies.
"""

import torch
from typing import Dict, Any, List, Optional, Iterator
from torch.utils.data import DataLoader, Dataset
from .interfaces.dataset_manager_interface import (
    IDatasetManager, 
    IDataProcessor, 
    IDataValidator, 
    IDataAugmentation, 
    IDataIterator
)
from data.data_loader import load_dataset
from utils.logging_utils import setup_logger


class PreferenceDatasetManager(IDatasetManager):
    """
    Dataset manager for preference learning datasets.
    
    Handles loading and management of preference datasets for RLHF training.
    """
    
    def __init__(self, tokenizer: Any):
        """
        Initialize the dataset manager.
        
        Args:
            tokenizer: Tokenizer for text processing
        """
        self.tokenizer = tokenizer
        self.logger = setup_logger("dataset_manager")
    
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
        try:
            self.logger.info(f"Loading dataset: {config.get('name', 'unknown')}")
            
            # Load dataset using existing data loader
            dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=config)
            
            self.logger.info(f"Dataset loaded successfully: {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}") from e
    
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
        try:
            # Extract DataLoader configuration
            batch_size = int(config.get("batch_size", 2))
            num_workers = int(config.get("dataloader_num_workers", 2))
            pin_memory = config.get("pin_memory", True)
            shuffle = config.get("shuffle", True)
            
            self.logger.info(f"Creating DataLoader: batch_size={batch_size}, num_workers={num_workers}")
            
            # Create data processor for collation
            processor = PreferenceDataProcessor(self.tokenizer, config)
            
            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=processor.collate_batch
            )
            
            self.logger.info("DataLoader created successfully")
            return dataloader
            
        except Exception as e:
            self.logger.error(f"Failed to create DataLoader: {e}")
            raise ValueError(f"DataLoader creation failed: {e}") from e
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            dataset: The dataset to get info for
            
        Returns:
            Dictionary containing dataset information
        """
        try:
            info = {
                "size": len(dataset),
                "type": type(dataset).__name__
            }
            
            # Try to get additional info if available
            if hasattr(dataset, 'features'):
                info["features"] = list(dataset.features.keys())
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Could not get complete dataset info: {e}")
            return {"size": len(dataset), "type": type(dataset).__name__}


class PreferenceDataProcessor(IDataProcessor):
    """
    Data processor for preference learning data.
    
    Handles tokenization and batch processing for preference datasets.
    """
    
    def __init__(self, tokenizer: Any, config: Dict[str, Any]):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: Tokenizer for text processing
            config: Processing configuration
        """
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.get("max_seq_length", 512)
        self.logger = setup_logger("data_processor")
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of data.
        
        Args:
            batch: List of data items
            
        Returns:
            Processed batch as dictionary of tensors
        """
        try:
            # Extract texts from batch
            chosen_texts = [item.get('chosen_text', '') for item in batch]
            rejected_texts = [item.get('rejected_text', '') for item in batch]
            
            # Tokenize texts
            chosen_encoded = self.tokenize_texts(chosen_texts, self.config)
            rejected_encoded = self.tokenize_texts(rejected_texts, self.config)
            
            return {
                'chosen_input_ids': chosen_encoded['input_ids'],
                'chosen_attention_mask': chosen_encoded['attention_mask'],
                'rejected_input_ids': rejected_encoded['input_ids'],
                'rejected_attention_mask': rejected_encoded['attention_mask']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            raise RuntimeError(f"Batch processing failed: {e}") from e
    
    def tokenize_texts(self, texts: List[str], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of texts.
        
        Args:
            texts: List of texts to tokenize
            config: Tokenization configuration
            
        Returns:
            Dictionary containing tokenized inputs
        """
        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return encoded
            
        except Exception as e:
            self.logger.error(f"Failed to tokenize texts: {e}")
            raise RuntimeError(f"Tokenization failed: {e}") from e
    
    def collate_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of data items.
        
        Args:
            batch: List of data items
            
        Returns:
            Collated batch as dictionary of tensors
        """
        return self.process_batch(batch)


class PreferenceDataValidator(IDataValidator):
    """
    Data validator for preference learning data.
    
    Validates dataset structure and batch content for preference learning.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = setup_logger("data_validator")
        self.validation_errors = []
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate the dataset structure and content.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            self.validation_errors.clear()
            
            # Check dataset size
            if len(dataset) == 0:
                self.validation_errors.append("Dataset is empty")
                return False
            
            # Check sample data
            sample = dataset[0]
            required_fields = ['chosen_text', 'rejected_text']
            
            for field in required_fields:
                if field not in sample:
                    self.validation_errors.append(f"Missing required field: {field}")
                    return False
                
                if not isinstance(sample[field], str) or len(sample[field].strip()) == 0:
                    self.validation_errors.append(f"Invalid {field}: must be non-empty string")
                    return False
            
            self.logger.info(f"Dataset validation passed: {len(dataset)} examples")
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Dataset validation error: {e}")
            self.logger.error(f"Dataset validation failed: {e}")
            return False
    
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        Validate a batch of data.
        
        Args:
            batch: Batch to validate
            
        Returns:
            True if batch is valid, False otherwise
        """
        try:
            self.validation_errors.clear()
            
            # Check required fields
            required_fields = ['chosen_input_ids', 'chosen_attention_mask', 
                             'rejected_input_ids', 'rejected_attention_mask']
            
            for field in required_fields:
                if field not in batch:
                    self.validation_errors.append(f"Missing required field: {field}")
                    return False
                
                if not isinstance(batch[field], torch.Tensor):
                    self.validation_errors.append(f"Invalid {field}: must be torch.Tensor")
                    return False
            
            # Check batch consistency
            chosen_ids = batch['chosen_input_ids']
            chosen_mask = batch['chosen_attention_mask']
            rejected_ids = batch['rejected_input_ids']
            rejected_mask = batch['rejected_attention_mask']
            
            if chosen_ids.shape != chosen_mask.shape:
                self.validation_errors.append("chosen_input_ids and chosen_attention_mask shape mismatch")
                return False
            
            if rejected_ids.shape != rejected_mask.shape:
                self.validation_errors.append("rejected_input_ids and rejected_attention_mask shape mismatch")
                return False
            
            if chosen_ids.shape[0] != rejected_ids.shape[0]:
                self.validation_errors.append("chosen and rejected batch size mismatch")
                return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Batch validation error: {e}")
            self.logger.error(f"Batch validation failed: {e}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()


class NoOpDataAugmentation(IDataAugmentation):
    """
    No-operation data augmentation.
    
    Placeholder implementation that doesn't modify data.
    """
    
    def __init__(self):
        """Initialize no-op augmentation."""
        self.enabled = False
        self.logger = setup_logger("data_augmentation")
    
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to a batch (no-op)."""
        return batch
    
    def is_augmentation_enabled(self) -> bool:
        """Check if augmentation is enabled."""
        return self.enabled
    
    def configure_augmentation(self, config: Dict[str, Any]) -> None:
        """Configure augmentation parameters."""
        self.enabled = config.get("enabled", False)
        self.logger.info(f"Data augmentation configured: enabled={self.enabled}")


class StandardDataIterator(IDataIterator):
    """
    Standard data iterator implementation.
    
    Provides standard iteration over DataLoader with optional customizations.
    """
    
    def __init__(self):
        """Initialize the data iterator."""
        self.batch_size = 1
        self.logger = setup_logger("data_iterator")
    
    def create_iterator(self, dataloader: DataLoader, config: Dict[str, Any]) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Create a custom iterator for the dataloader.
        
        Args:
            dataloader: DataLoader to iterate over
            config: Iterator configuration
            
        Returns:
            Custom iterator
        """
        self.batch_size = dataloader.batch_size
        self.logger.info(f"Created iterator with batch size: {self.batch_size}")
        return iter(dataloader)
    
    def get_batch_size(self) -> int:
        """Get the current batch size."""
        return self.batch_size
    
    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size."""
        self.batch_size = batch_size
        self.logger.info(f"Batch size set to: {batch_size}")


class DatasetManagerFactory:
    """
    Factory for creating dataset manager components.
    
    Provides a centralized way to create different types of dataset managers.
    """
    
    @staticmethod
    def create_preference_manager(tokenizer: Any) -> IDatasetManager:
        """Create a preference dataset manager."""
        return PreferenceDatasetManager(tokenizer)
    
    @staticmethod
    def create_data_processor(tokenizer: Any, config: Dict[str, Any]) -> IDataProcessor:
        """Create a data processor."""
        return PreferenceDataProcessor(tokenizer, config)
    
    @staticmethod
    def create_data_validator() -> IDataValidator:
        """Create a data validator."""
        return PreferenceDataValidator()
    
    @staticmethod
    def create_data_augmentation(config: Dict[str, Any]) -> IDataAugmentation:
        """Create a data augmentation strategy."""
        return NoOpDataAugmentation()
    
    @staticmethod
    def create_data_iterator() -> IDataIterator:
        """Create a data iterator."""
        return StandardDataIterator()
