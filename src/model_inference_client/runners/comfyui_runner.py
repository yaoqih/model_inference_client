"""
Concrete implementation of the BaseRunner for ComfyUI.
"""
import asyncio
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

import httpx
from model_inference_client.models.schemas import ModelInstanceInfo, ModelInstanceStatus
from model_inference_client.runners.base_runner import BaseRunner
from model_inference_client.utils.network_utils import find_free_port


class ComfyUIRunner(BaseRunner):
    """
    A runner for managing ComfyUI server processes.
    """

    def __init__(self, runner_config: Dict[str, Any]):
        super().__init__(runner_config)
        self._lock = asyncio.Lock()
        self.instances: Dict[str, ModelInstanceInfo] = {}
        print("ComfyUIRunner initialized")

    async def _wait_for_service(self, port: int, timeout: int = 60):
        """Waits for the ComfyUI service to be available."""
        start_time = time.time()
        url = f"http://127.0.0.1:{port}/history"
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"ComfyUI service on port {port} is ready.")
                        return
                except httpx.RequestError:
                    await asyncio.sleep(1)
                else:
                    await asyncio.sleep(1)
        raise TimeoutError(f"ComfyUI service on port {port} did not become ready in {timeout} seconds.")

    async def start(
        self,
        model_name: str,
        gpu_id: int,
        **kwargs,
    ) -> ModelInstanceInfo:
        """
        Starts a ComfyUI server instance.
        """
        async with self._lock:
            instance_id = str(uuid.uuid4())
            port = find_free_port()

            comfyui_path = self.config.get("comfyui_path")
            if not comfyui_path or not os.path.exists(comfyui_path):
                raise FileNotFoundError("ComfyUI path not configured or does not exist.")

            cmd = [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=comfyui_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            await self._wait_for_service(port)

            instance_info = ModelInstanceInfo(
                instance_id=instance_id,
                model_name=model_name,
                model_type="comfyui",
                status=ModelInstanceStatus.RUNNING,
                port=port,
                pid=process.pid,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                active_gpu_ids=[gpu_id],
            )
            self.instances[instance_id] = instance_info
            print(f"Started ComfyUI for model {model_name} on GPU {gpu_id} with PID {process.pid}")
            return instance_info

    async def stop(self, **kwargs) -> Optional[ModelInstanceInfo]:
        """
        Stops a ComfyUI server instance.
        """
        instance_id = kwargs.get("instance_id")
        if not instance_id:
            # For ComfyUI, we might need to find the instance by model_name and gpu_id
            model_name = kwargs.get("model_name")
            gpu_id = kwargs.get("gpu_id")
            for inst_id, info in self.instances.items():
                if info.model_name == model_name and gpu_id in info.active_gpu_ids:
                    instance_id = inst_id
                    break
        
        if not instance_id or instance_id not in self.instances:
            print(f"ComfyUI instance {instance_id} not found.")
            return None

        async with self._lock:
            instance = self.instances.pop(instance_id)
            try:
                os.kill(instance.pid, signal.SIGTERM)
                print(f"Sent SIGTERM to ComfyUI instance {instance_id} with PID {instance.pid}")
            except ProcessLookupError:
                print(f"Process with PID {instance.pid} not found for instance {instance_id}.")
            
            instance.status = ModelInstanceStatus.STOPPED
            instance.updated_at = datetime.utcnow().isoformat()
            return instance

    async def get_status(self, model_name: str) -> Optional[ModelInstanceInfo]:
        """
        Gets the status of a ComfyUI server instance by model name.
        Assumes one instance per model name for simplicity.
        """
        async with self._lock:
            for instance in self.instances.values():
                if instance.model_name == model_name:
                    # Check if process is still alive
                    try:
                        os.kill(instance.pid, 0)
                    except OSError:
                        instance.status = ModelInstanceStatus.STOPPED
                    return instance
            return None

    async def get_all_active_models(self) -> List[ModelInstanceInfo]:
        """Returns a list of all active model instances managed by this runner."""
        async with self._lock:
            return list(self.instances.values())

    async def shutdown(self):
        """Stops all managed ComfyUI instances."""
        print("Shutting down all ComfyUI instances...")
        async with self._lock:
            for instance_id in list(self.instances.keys()):
                await self.stop(instance_id=instance_id)
        print("All ComfyUI instances have been shut down.")
