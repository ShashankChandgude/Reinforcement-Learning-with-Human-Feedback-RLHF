"""
Weights & Biases integration for experiment tracking.
Import is optional; if wandb is not installed, logging is skipped gracefully.
"""
import os
from typing import Dict, Any, Optional
from utils.logging_utils import setup_logger

# Try to import wandb; fall back to a stub when unavailable
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


class WandBLogger:
    """Weights & Biases logger for experiment tracking."""
    
    def __init__(self, project_name: str = "rlhf-pipeline", config: Optional[Dict[str, Any]] = None):
        self.project_name = project_name
        self.config = config or {}
        self.logger = setup_logger("wandb")
        self.is_initialized = False
    
    def init(self, run_name: Optional[str] = None, tags: Optional[list] = None):
        """Initialize W&B run."""
        # Check if W&B is available
        if not self._check_wandb_available():
            self.logger.warning("W&B not available, skipping initialization")
            return False

        try:
            # Initialize run
            wandb.init(
                project=self.project_name,
                name=run_name,
                config=self.config,
                tags=tags or [],
                reinit=True
            )
            self.is_initialized = True
            self.logger.info(f"W&B initialized for project: {self.project_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize W&B: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if not self.is_initialized:
            return
        
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """Log histogram to W&B."""
        if not self.is_initialized:
            return
        
        try:
            wandb.log({name: wandb.Histogram(values)}, step=step)
        except Exception as e:
            self.logger.error(f"Failed to log histogram: {e}")
    
    def log_table(self, name: str, columns: list, data: list):
        """Log table to W&B."""
        if not self.is_initialized:
            return
        
        try:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({name: table})
        except Exception as e:
            self.logger.error(f"Failed to log table: {e}")
    
    def log_model_artifacts(self, model_dir: str, artifact_name: str):
        """Log model artifacts to W&B."""
        if not self.is_initialized:
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_dir(model_dir)
            wandb.log_artifact(artifact)
            self.logger.info(f"Model artifacts logged: {artifact_name}")
        except Exception as e:
            self.logger.error(f"Failed to log model artifacts: {e}")
    
    def log_training_curves(self, training_history: Dict[str, list]):
        """Log training curves to W&B."""
        if not self.is_initialized:
            return
        
        try:
            # Create plots for each metric
            for metric_name, values in training_history.items():
                if values:
                    # Log as line plot
                    wandb.log({f"training/{metric_name}": wandb.plot.line(
                        x=list(range(len(values))),
                        y=values,
                        title=f"Training {metric_name}",
                        xname="Step"
                    )})
            
            self.logger.info("Training curves logged to W&B")
        except Exception as e:
            self.logger.error(f"Failed to log training curves: {e}")
    
    def log_sample_generations(self, prompts: list, responses: list, step: Optional[int] = None):
        """Log sample generations to W&B."""
        if not self.is_initialized:
            return
        
        try:
            # Create table with sample generations
            table_data = []
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                table_data.append([
                    i,
                    prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    response[:200] + "..." if len(response) > 200 else response
                ])
            
            self.log_table(
                name="sample_generations",
                columns=["Index", "Prompt", "Response"],
                data=table_data
            )
            
            self.logger.info(f"Sample generations logged to W&B")
        except Exception as e:
            self.logger.error(f"Failed to log sample generations: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if self.is_initialized:
            try:
                wandb.finish()
                self.logger.info("W&B run finished")
            except Exception as e:
                self.logger.error(f"Failed to finish W&B run: {e}")
    
    def _check_wandb_available(self) -> bool:
        """Check if W&B is available and configured."""
        if not _WANDB_AVAILABLE:
            self.logger.warning("W&B not installed. Install with: pip install wandb")
            return False
        try:
            if not wandb.api.api_key:
                self.logger.warning("W&B API key not found. Please run 'wandb login'")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error checking W&B availability: {e}")
            return False


def setup_wandb_logging(config: Dict[str, Any], run_name: Optional[str] = None) -> WandBLogger:
    """Setup W&B logging with given configuration."""
    logger = WandBLogger(
        project_name=config.get("project_name", "rlhf-pipeline"),
        config=config
    )
    
    tags = config.get("tags", [])
    if config.get("use_lora", False):
        tags.append("lora")
    if config.get("use_ppo", False):
        tags.append("ppo")
    
    logger.init(run_name=run_name, tags=tags)
    return logger


# Example usage
if __name__ == "__main__":
    # Test W&B logging
    config = {
        "model": "test-model",
        "learning_rate": 1e-5,
        "batch_size": 4,
        "project_name": "rlhf-test"
    }
    
    wandb_logger = setup_wandb_logging(config, run_name="test-run")
    
    # Log some test metrics
    wandb_logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
    wandb_logger.log_metrics({"loss": 0.4, "accuracy": 0.85}, step=2)
    
    # Log sample generations
    wandb_logger.log_sample_generations(
        prompts=["Test prompt 1", "Test prompt 2"],
        responses=["Test response 1", "Test response 2"]
    )
    
    wandb_logger.finish()
