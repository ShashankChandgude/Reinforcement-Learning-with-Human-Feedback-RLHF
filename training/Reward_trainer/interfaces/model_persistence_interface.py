"""
Model Persistence Interface

Defines the contract for model saving and loading in RLHF training.
Supports different persistence strategies and formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
from torch.nn import Module
import os


class IModelPersistence(ABC):
    """Interface for model persistence operations."""
    
    @abstractmethod
    def save_model(self, 
                  model: Module, 
                  path: str, 
                  config: Dict[str, Any],
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
            config: Model configuration
            metadata: Optional metadata to save
            
        Raises:
            ValueError: If path is invalid
            RuntimeError: If saving fails
        """
        pass
    
    @abstractmethod
    def load_model(self, 
                  path: str, 
                  config: Dict[str, Any]) -> Module:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            config: Model configuration
            
        Returns:
            Loaded model
            
        Raises:
            FileNotFoundError: If model file is not found
            RuntimeError: If loading fails
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, 
                       model: Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       path: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            path: Path to save checkpoint
            metadata: Optional metadata
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, 
                       path: str, 
                       model: Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Checkpoint metadata
        """
        pass


class IConfigurationPersistence(ABC):
    """Interface for configuration persistence."""
    
    @abstractmethod
    def save_configuration(self, 
                          config: Dict[str, Any], 
                          path: str) -> None:
        """
        Save configuration to disk.
        
        Args:
            config: Configuration dictionary
            path: Path to save configuration
        """
        pass
    
    @abstractmethod
    def load_configuration(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from disk.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        pass


class IMetadataManager(ABC):
    """Interface for metadata management."""
    
    @abstractmethod
    def create_metadata(self, 
                       model: Module, 
                       config: Dict[str, Any],
                       training_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create metadata for a model.
        
        Args:
            model: Model to create metadata for
            config: Model configuration
            training_info: Optional training information
            
        Returns:
            Metadata dictionary
        """
        pass
    
    @abstractmethod
    def save_metadata(self, 
                     metadata: Dict[str, Any], 
                     path: str) -> None:
        """
        Save metadata to disk.
        
        Args:
            metadata: Metadata dictionary
            path: Path to save metadata
        """
        pass
    
    @abstractmethod
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """
        Load metadata from disk.
        
        Args:
            path: Path to metadata file
            
        Returns:
            Loaded metadata dictionary
        """
        pass


class IFileManager(ABC):
    """Interface for file management operations."""
    
    @abstractmethod
    def create_directory(self, path: str) -> None:
        """
        Create a directory if it doesn't exist.
        
        Args:
            path: Directory path to create
        """
        pass
    
    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: File path to check
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    def list_files(self, directory: str, pattern: Optional[str] = None) -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list
            pattern: Optional file pattern to match
            
        Returns:
            List of file paths
        """
        pass
    
    @abstractmethod
    def cleanup_old_files(self, directory: str, max_files: int) -> None:
        """
        Clean up old files in a directory.
        
        Args:
            directory: Directory to clean up
            max_files: Maximum number of files to keep
        """
        pass


class IModelValidator(ABC):
    """Interface for model validation."""
    
    @abstractmethod
    def validate_model_structure(self, model: Module) -> bool:
        """
        Validate model structure.
        
        Args:
            model: Model to validate
            
        Returns:
            True if model structure is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_model_weights(self, model: Module) -> bool:
        """
        Validate model weights.
        
        Args:
            model: Model to validate
            
        Returns:
            True if model weights are valid, False otherwise
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
