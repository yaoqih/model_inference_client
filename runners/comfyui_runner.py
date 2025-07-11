"""
Concrete implementation of the BaseRunner for ComfyUI.
"""
from typing import Dict, Any

from ..models.schemas import ModelInstanceInfo
from .base_runner import BaseRunner


class ComfyUIRunner(BaseRunner):
    """
    A runner for managing ComfyUI server processes.
    """

    def __init__(self, runner_config: Dict[str, Any]):
        super().__init__(runner_config)
        # TODO: Initialize any ComfyUI specific settings, e.g., path to executable
        print("ComfyUIRunner initialized")

    def start(
        self,
        model_name: str,
        gpu_id: int,
        model_path: str = None,
        **kwargs,
    ) -> ModelInstanceInfo:
        """
        Starts a ComfyUI server instance.
        """
        # TODO: Implement the logic to start the ComfyUI process
        # 1. Construct the command line arguments (e.g., --listen, --port)
        # 2. Use subprocess.Popen to launch the process with appropriate CUDA_VISIBLE_DEVICES
        # 3. Capture PID, find a free port
        # 4. Return a ModelInstanceInfo object
        print(f"Starting ComfyUI for model {model_name} on GPU {gpu_id}")
        pass

    def stop(self, instance: ModelInstanceInfo) -> None:
        """
        Stops a ComfyUI server instance.
        """
        # TODO: Implement the logic to stop the ComfyUI process
        # 1. Find the process by instance_id (from a state manager)
        # 2. Terminate the process using its PID
        print(f"Stopping ComfyUI instance {instance.instance_id}")
        pass

    def get_status(self, instance_id: str) -> ModelInstanceInfo:
        """
        Gets the status of a ComfyUI server instance.
        """
        # TODO: Implement the logic to check the process status
        # 1. Find the process by instance_id
        # 2. Check if the process is alive
        # 3. Potentially ping a health check endpoint on the server
        print(f"Getting status for ComfyUI instance {instance_id}")
        pass
