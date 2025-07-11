"""
Abstract Base Class for model runners.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..models.schemas import ModelInstanceInfo


class BaseRunner(ABC):
    """
    An abstract base class for all model serving runners.

    This class defines the common interface for starting, stopping, and checking
    the status of a model serving process.
    """

    def __init__(self, runner_config: Dict[str, Any]):
        """
        Initializes the runner with backend-specific configuration.

        Args:
            runner_config: A dictionary containing configuration for the runner.
        """
        self.config = runner_config

    @abstractmethod
    def start(
        self,
        model_name: str,
        gpu_id: int,
        model_path: str = None,
        **kwargs,
    ) -> ModelInstanceInfo:
        """
        Starts a model serving process.

        This method should handle the specifics of launching the serving
        executable (e.g., tritonserver, ComfyUI's main.py) as a subprocess.
        It must return a ModelInstanceInfo object containing the process PID,
        port, and other relevant details.

        Args:
            model_name: The name of the model to serve.
            gpu_id: The GPU device ID to use.
            model_path: Optional path to the model's data.
            **kwargs: Additional backend-specific arguments.

        Returns:
            A ModelInstanceInfo object representing the newly started instance.
        """
        pass

    @abstractmethod
    def stop(self, instance: ModelInstanceInfo) -> None:
        """
        Stops a model serving process.

        Args:
            instance: The full info object of the model instance to stop.
        """
        pass

    @abstractmethod
    def get_status(self, instance_id: str) -> ModelInstanceInfo:
        """
        Gets the status of a model serving process.

        Args:
            instance_id: The unique ID of the model instance to check.

        Returns:
            A ModelInstanceInfo object with the current status.
        """
        pass
