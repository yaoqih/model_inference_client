"""
Concrete implementation of the BaseRunner for Triton Inference Server.
This runner manages a single, long-lived Triton Docker container and
controls the models loaded within it via Triton's API.
"""
import time
import uuid
import subprocess
import json
import requests
from datetime import datetime
from threading import Lock
from typing import Dict, Any, Optional, Set, List
from pathlib import Path

from tritonclient.http import InferenceServerClient

from ..models.schemas import ModelInstanceInfo, ModelInstanceStatus
from .base_runner import BaseRunner
from ..utils.config_utils import generate_triton_config, remove_triton_config


class TritonRunner(BaseRunner):
    """
    Manages a single Triton Inference Server Docker container and
    tracks which models are running on which GPUs.
    """

    def __init__(self, runner_config: Dict[str, Any]):
        super().__init__(runner_config)
        self.triton_client: Optional[InferenceServerClient] = None
        self._lock = Lock()

        # State management - track which GPUs each model is using
        self.container_id: Optional[str] = None
        self.model_gpu_mapping: Dict[str, Set[int]] = {}  # model_name -> set of gpu_ids
        self.model_info: Dict[str, ModelInstanceInfo] = {}  # model_name -> info
        self.server_is_ready = False

        self._find_existing_container()
        print("TritonRunner initialized.")

    def _run_command(self, cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
        """Execute a shell command and return the result."""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=capture_output, 
                text=True, 
                check=True
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            raise e

    def _http_request(self, method: str, endpoint: str, data: Optional[Dict] = None, timeout: int = 180) -> requests.Response:
        """Make HTTP request to Triton server with proper error handling."""
        url = f"http://localhost:{self.config['http_port']}/{endpoint}"
        
        try:
            response = requests.request(
                method=method,
                url=url, 
                json=data,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            return response
        except requests.exceptions.Timeout:
            raise TimeoutError(f"HTTP {method} request to {endpoint} timed out")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Failed to connect to Triton server at {url}")
        except Exception as e:
            raise RuntimeError(f"HTTP {method} request failed: {e}")

    def _load_model_http(self, model_name: str) -> bool:
        """Load model using HTTP API (non-blocking)."""
        try:
            response = self._http_request("POST", f"v2/repository/models/{model_name}/load")
            if response.status_code == 200:
                print(f"Successfully initiated loading of model '{model_name}'")
                return True
            else:
                print(f"Failed to load model '{model_name}': {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return False

    def _unload_model_http(self, model_name: str) -> bool:
        """Unload model using HTTP API (non-blocking)."""
        try:
            response = self._http_request("POST", f"v2/repository/models/{model_name}/unload")
            if response.status_code == 200:
                print(f"Successfully initiated unloading of model '{model_name}'")
                return True
            else:
                print(f"Failed to unload model '{model_name}': {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error unloading model '{model_name}': {e}")
            return False

    def _wait_for_model_ready(self, model_name: str, timeout: int = 60) -> bool:
        """Wait for model to become ready using HTTP polling."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._http_request("GET", f"v2/models/{model_name}/ready", timeout=5)
                if response.status_code == 200:
                    return True
            except Exception:
                pass  # Continue polling
            time.sleep(1)
        return False

    def _is_model_ready_http(self, model_name: str) -> bool:
        """Check if model is ready using HTTP API."""
        try:
            response = self._http_request("GET", f"v2/models/{model_name}/ready", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _find_existing_container(self):
        """Finds if the managed container is already running on startup."""
        try:
            container_name = self.config["container_name"]
            # Check if container exists and is running
            result = self._run_command([
                "docker", "ps", "--filter", f"name={container_name}", 
                "--format", "{{.ID}}\t{{.Status}}"
            ])
            
            if result.stdout.strip():
                container_info = result.stdout.strip().split('\t')
                self.container_id = container_info[0]
                status = container_info[1]
                if "Up" in status:
                    self._on_server_started()
                    print(f"Found existing running Triton container '{container_name}' (ID: {self.container_id}).")
                else:
                    print(f"Found Triton container '{container_name}' but it's not running (Status: {status}).")
                    self.container_id = None
            else:
                self.container_id = None
                print("No existing Triton container found.")
        except subprocess.CalledProcessError:
            self.container_id = None
            print("No existing Triton container found.")
        except Exception as e:
            self.container_id = None
            print(f"Error checking for existing container: {e}")

    def _start_server(self) -> str:
        """Starts the Triton Docker container using command line."""
        if self.container_id:
            return self.container_id

        print("Starting Triton Inference Server container...")
        
        # Build the docker run command
        cmd = [
            "docker", "run", "-d", "--rm", "--shm-size=2g", "--net=host",
            "--name", self.config["container_name"],
            "-v", f"{self.config['model_repository_host_path']}:/models",
            "-v", f"{self.config['runtime_host_path']}:/runtime"
        ]
        
        # Add GPU support if configured
        if self.config.get("gpus"):
            cmd.extend(["--gpus", "all"])
            
        # Add the image name
        cmd.append(self.config["image_name"])
        
        # Add tritonserver command and arguments
        cmd.extend([
            "tritonserver",
            "--model-repository=/models",
            "--model-control-mode=explicit",
            f"--http-port={self.config['http_port']}",
            f"--grpc-port={self.config['grpc_port']}",
            f"--metrics-port={self.config['metrics_port']}"
        ])
        
        # Execute the command
        result = self._run_command(cmd)
        self.container_id = result.stdout.strip()
        
        print(f"Container started with ID: {self.container_id}")
        self._wait_for_server_ready()
        self._on_server_started()
        return self.container_id
    
    def _on_server_started(self):
        """Actions to perform once the server container is confirmed running."""
        self.triton_client = InferenceServerClient(
            url=f"localhost:{self.config['http_port']}",
            verbose=False
        )
        self.server_is_ready = True
        print("Triton server is ready.")
        
        # Sync existing model states if container was already running
        self._sync_existing_models()

    def _sync_existing_models(self):
        """Synchronize internal state with models currently loaded in Triton."""
        if not self.server_is_ready:
            return
        
        try:
            # Get list of all models in the repository using HTTP (POST method required)
            response = self._http_request("POST", "v2/repository/index", timeout=10)
            if response.status_code != 200:
                print(f"Failed to get repository index: {response.status_code}")
                return
                
            repository_data = response.json()
            
            for model_info in repository_data:
                model_name = model_info['name']
                model_state = model_info.get('state', 'UNKNOWN')
                
                # Only sync models that are currently ready/loaded
                if model_state == 'READY' and self._is_model_ready_http(model_name):
                    print(f"Syncing existing model '{model_name}' state...")
                    
                    # Parse the current config to extract GPU assignments
                    gpu_ids = self._extract_gpu_ids_from_current_config(model_name)
                    
                    if gpu_ids:
                        # Update our internal state
                        self.model_gpu_mapping[model_name] = set(gpu_ids)
                        
                        # Create ModelInstanceInfo for this model
                        instances_per_gpu = self._get_instances_per_gpu(model_name)
                        current_time = datetime.utcnow().isoformat()
                        
                        self.model_info[model_name] = ModelInstanceInfo(
                            model_name=model_name,
                            model_type="triton",
                            active_gpu_ids=sorted(gpu_ids),
                            total_instances=len(gpu_ids) * instances_per_gpu,
                            status=ModelInstanceStatus.RUNNING,
                            pid=self._get_container_pid(),
                            port=self.config["http_port"],
                            created_at=current_time,
                            updated_at=current_time,
                        )
                        
                        print(f"Synced model '{model_name}' running on GPUs: {gpu_ids}")
                    else:
                        print(f"Model '{model_name}' is running but no GPU assignment found in config")
                    
        except Exception as e:
            print(f"Warning: Failed to sync existing models: {e}")

    def _extract_gpu_ids_from_current_config(self, model_name: str) -> List[int]:
        """Extract GPU IDs from the current model configuration."""
        try:
            # Read the current config.pbtxt file
            config_path = Path(self.config["model_repository_host_path"]) / model_name / "config.pbtxt"
            if not config_path.exists():
                return []
            
            config_content = config_path.read_text()
            gpu_ids = []
            
            # Parse instance_group sections to extract GPU IDs
            lines = config_content.split('\n')
            in_instance_group = False
            in_gpu_section = False
            
            for line in lines:
                line = line.strip()
                
                if 'instance_group' in line and '[' in line:
                    in_instance_group = True
                    continue
                    
                if in_instance_group and line == ']':
                    in_instance_group = False
                    continue
                    
                if in_instance_group and 'gpus:' in line:
                    in_gpu_section = True
                    # Extract GPU IDs from the same line if present
                    if '[' in line and ']' in line:
                        gpu_part = line.split('[')[1].split(']')[0]
                        gpu_ids.extend([int(x.strip()) for x in gpu_part.split(',') if x.strip().isdigit()])
                        in_gpu_section = False
                    continue
                    
                if in_gpu_section and '[' in line and ']' in line:
                    gpu_part = line.split('[')[1].split(']')[0]
                    gpu_ids.extend([int(x.strip()) for x in gpu_part.split(',') if x.strip().isdigit()])
                    in_gpu_section = False
                    
            return gpu_ids
            
        except Exception as e:
            print(f"Error extracting GPU IDs from config for {model_name}: {e}")
            return []

    def _get_container_pid(self) -> Optional[int]:
        """Get the PID of the current container."""
        if not self.container_id:
            return None
        
        try:
            result = self._run_command(["docker", "inspect", self.container_id, "--format", "{{.State.Pid}}"])
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None

    def _wait_for_server_ready(self):
        """Waits until the Triton server is live and ready."""
        print("Waiting for Triton server to be ready...")
        timeout = self.config.get("api_timeout_seconds", 120)
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Use HTTP to check if server is live
                response = requests.get(
                    f"http://localhost:{self.config['http_port']}/v2/health/live",
                    timeout=5
                )
                if response.status_code == 200:
                    print("Triton server is live.")
                    return
            except Exception:
                pass # Ignore connection errors while waiting
            time.sleep(1)
        raise TimeoutError("Triton server failed to become ready in time.")

    def _get_instances_per_gpu(self, model_name: str) -> int:
        """Get the configured instances per GPU for a model."""
        model_configs = self.config.get("model_gpu_configs", {})
        if model_name in model_configs:
            return model_configs[model_name]["instances_per_gpu"]
        return model_configs.get("default", {}).get("instances_per_gpu", 1)

    def _update_model_config_and_reload(self, model_name: str):
        """Regenerate config and reload the model in Triton."""
        gpu_ids = list(self.model_gpu_mapping.get(model_name, set()))
        
        if not gpu_ids:
            # No GPUs assigned, remove config and unload model
            if self.triton_client and self.server_is_ready:
                try:
                    if self._is_model_ready_http(model_name):
                        print(f"Unloading model '{model_name}' from Triton...")
                        self._unload_model_http(model_name)
                except Exception as e:
                    print(f"Error unloading model '{model_name}': {e}")
            
            remove_triton_config(model_name, self.config["model_repository_host_path"])
            return

        # Generate new config with current GPU assignment
        instances_per_gpu = self._get_instances_per_gpu(model_name)
        config_generated = generate_triton_config(
            model_name=model_name,
            model_repository_path=self.config["model_repository_host_path"],
            gpu_ids=gpu_ids,
            instance_count_per_gpu=instances_per_gpu
        )
        
        if not config_generated:
            raise FileNotFoundError(f"Could not generate config for {model_name}. Check for a template.")

        # Load/reload the model in Triton using non-blocking HTTP request
        if self.triton_client and self.server_is_ready:
            try:
                print(f"Initiating load/reload of model '{model_name}' with new configuration...")
                success = self._load_model_http(model_name)
                if not success:
                    raise RuntimeError(f"Failed to initiate loading of model '{model_name}'")
                
                print(f"Model '{model_name}' load request sent successfully. Model will be available shortly.")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to reload model '{model_name}': {e}")

    def _update_model_status(self, model_name: str):
        """Update model status from STARTING to RUNNING if model is ready."""
        if model_name in self.model_info and self.model_info[model_name].status == ModelInstanceStatus.STARTING:
            if self._is_model_ready_http(model_name):
                self.model_info[model_name].status = ModelInstanceStatus.RUNNING
                self.model_info[model_name].updated_at = datetime.utcnow().isoformat()

    def shutdown_server_if_needed(self):
        """Stops the Triton container if no models are active."""
        with self._lock:
            if self.container_id and not self.model_gpu_mapping:
                print("No active models. Shutting down Triton server.")
                try:
                    # Stop and remove the container
                    self._run_command(["docker", "stop", self.container_id])
                    print(f"Container {self.container_id} stopped.")
                except subprocess.CalledProcessError as e:
                    print(f"Error stopping container {self.container_id}: {e}")
                finally:
                    self.container_id = None
                    self.triton_client = None
                    self.server_is_ready = False

    def start(self, **kwargs) -> ModelInstanceInfo:
        """Add the specified GPU to a model's instance group."""
        model_name = kwargs["model_name"]
        gpu_id = kwargs["gpu_id"]

        with self._lock:
            # Ensure server is running
            if not self.container_id or not self.server_is_ready:
                self._start_server()

            # Initialize model tracking if this is the first time we see this model
            if model_name not in self.model_gpu_mapping:
                self.model_gpu_mapping[model_name] = set()
                current_time = datetime.utcnow().isoformat()
                self.model_info[model_name] = ModelInstanceInfo(
                    model_name=model_name,
                    model_type="triton",
                    active_gpu_ids=[],
                    total_instances=0,
                    status=ModelInstanceStatus.STARTING,
                    pid=None,
                    port=self.config["http_port"],
                    created_at=current_time,
                    updated_at=current_time,
                )

            # Check if GPU is already assigned to this model
            if gpu_id in self.model_gpu_mapping[model_name]:
                print(f"GPU {gpu_id} is already assigned to model '{model_name}'")
                return self.model_info[model_name]

            # Add the new GPU to the model's assignment
            original_gpus = self.model_gpu_mapping[model_name].copy()
            self.model_gpu_mapping[model_name].add(gpu_id)
            
            try:
                # Update config and reload model
                self._update_model_config_and_reload(model_name)
                
                # Update model info - mark as starting since load is async
                instances_per_gpu = self._get_instances_per_gpu(model_name)
                self.model_info[model_name].active_gpu_ids = sorted(list(self.model_gpu_mapping[model_name]))
                self.model_info[model_name].total_instances = len(self.model_gpu_mapping[model_name]) * instances_per_gpu
                self.model_info[model_name].status = ModelInstanceStatus.STARTING  # Keep as STARTING since load is async
                self.model_info[model_name].updated_at = datetime.utcnow().isoformat()
                
                # Get container PID
                if self.container_id:
                    try:
                        result = self._run_command(["docker", "inspect", self.container_id, "--format", "{{.State.Pid}}"])
                        self.model_info[model_name].pid = int(result.stdout.strip())
                    except (subprocess.CalledProcessError, ValueError):
                        self.model_info[model_name].pid = None

                print(f"Successfully initiated start of model '{model_name}' on GPU {gpu_id}")
                return self.model_info[model_name]
                
            except Exception as e:
                # Rollback GPU assignment on failure
                self.model_gpu_mapping[model_name] = original_gpus
                if not original_gpus:
                    # If this was the first assignment and it failed, clean up
                    del self.model_gpu_mapping[model_name]
                    del self.model_info[model_name]
                
                self.shutdown_server_if_needed()
                raise e

    def stop(self, **kwargs) -> Optional[ModelInstanceInfo]:
        """Remove the specified GPU from a model's instance group."""
        model_name = kwargs["model_name"]
        gpu_id = kwargs["gpu_id"]

        with self._lock:
            if model_name not in self.model_gpu_mapping:
                print(f"Model '{model_name}' is not currently running.")
                return None

            # Check if the GPU is actually assigned to this model
            if gpu_id not in self.model_gpu_mapping[model_name]:
                print(f"GPU {gpu_id} is not assigned to model '{model_name}'")
                return self.model_info[model_name]

            # Remove the specified GPU
            self.model_gpu_mapping[model_name].discard(gpu_id)
            
            try:
                # Update config and reload (or unload if no GPUs left)
                self._update_model_config_and_reload(model_name)
                
                if not self.model_gpu_mapping[model_name]:
                    # No GPUs left, remove model completely
                    del self.model_gpu_mapping[model_name]
                    final_info = self.model_info[model_name]
                    final_info.status = ModelInstanceStatus.STOPPED
                    final_info.active_gpu_ids = []
                    final_info.total_instances = 0
                    final_info.updated_at = datetime.utcnow().isoformat()
                    del self.model_info[model_name]
                    
                    print(f"Model '{model_name}' completely stopped.")
                    self.shutdown_server_if_needed()
                    return final_info
                else:
                    # Update model info with remaining GPUs
                    instances_per_gpu = self._get_instances_per_gpu(model_name)
                    self.model_info[model_name].active_gpu_ids = sorted(list(self.model_gpu_mapping[model_name]))
                    self.model_info[model_name].total_instances = len(self.model_gpu_mapping[model_name]) * instances_per_gpu
                    self.model_info[model_name].updated_at = datetime.utcnow().isoformat()
                    
                    print(f"Removed GPU {gpu_id} from model '{model_name}'. Still running on {self.model_info[model_name].active_gpu_ids}")
                    return self.model_info[model_name]
                    
            except Exception as e:
                print(f"Error stopping model '{model_name}' on GPU {gpu_id}: {e}")
                return None

    def get_all_active_models(self) -> List[ModelInstanceInfo]:
        """Returns a list of all active model instances managed by this runner."""
        with self._lock:
            # Update status of any starting models
            for model_name in list(self.model_info.keys()):
                self._update_model_status(model_name)
            return list(self.model_info.values())

    def get_status(self, model_name: str) -> Optional[ModelInstanceInfo]:
        """Get the status of a specific model."""
        with self._lock:
            if model_name in self.model_info:
                self._update_model_status(model_name)
                return self.model_info[model_name]
            return None

if __name__ == "__main__":
    # To run this test script, ensure you are in the project root directory
    # and execute: python -m model_inference_client.runners.triton_runner
    from ..config import TRITON_RUNNER_CONFIG
    runner = TritonRunner(TRITON_RUNNER_CONFIG)
    print(runner.get_all_active_models())