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

from models.schemas import ModelInstanceInfo, ModelInstanceStatus
from .base_runner import BaseRunner
from utils.config_utils import generate_triton_config, remove_triton_config
from utils.network_utils import find_free_port


class TritonRunner(BaseRunner):
    """
    Manages Triton Inference Server containers.
    Supports a single shared container or dedicated containers per model.
    """

    def __init__(self, runner_config: Dict[str, Any]):
        super().__init__(runner_config)
        self._lock = Lock()
        self.use_dedicated_server = self.config.get("use_dedicated_server", False)

        # State for shared server mode
        self.shared_container_id: Optional[str] = None
        self.shared_triton_client: Optional[InferenceServerClient] = None
        self.shared_server_is_ready = False

        # State for dedicated server mode
        self.dedicated_containers: Dict[str, Dict[str, Any]] = {}  # model_name -> {container_id, http_port, ...}

        # Common state
        self.model_gpu_mapping: Dict[str, Set[int]] = {}  # model_name -> set of gpu_ids
        self.model_info: Dict[str, ModelInstanceInfo] = {}  # model_name -> info

        if not self.use_dedicated_server:
            self._find_existing_container()
        
        print(f"TritonRunner initialized in {'dedicated' if self.use_dedicated_server else 'shared'} server mode.")

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

    def _http_request(self, method: str, endpoint: str, http_port: int, data: Optional[Dict] = None, timeout: int = 180) -> requests.Response:
        """Make HTTP request to a Triton server with proper error handling."""
        url = f"http://localhost:{http_port}/{endpoint}"
        
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

    def _load_model_http(self, model_name: str, http_port: int) -> bool:
        """Load model using HTTP API (non-blocking)."""
        try:
            response = self._http_request("POST", f"v2/repository/models/{model_name}/load", http_port)
            if response.status_code == 200:
                print(f"Successfully initiated loading of model '{model_name}' on port {http_port}")
                return True
            else:
                print(f"Failed to load model '{model_name}': {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return False

    def _unload_model_http(self, model_name: str, http_port: int) -> bool:
        """Unload model using HTTP API (non-blocking)."""
        try:
            response = self._http_request("POST", f"v2/repository/models/{model_name}/unload", http_port)
            if response.status_code == 200:
                print(f"Successfully initiated unloading of model '{model_name}' on port {http_port}")
                return True
            else:
                print(f"Failed to unload model '{model_name}': {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error unloading model '{model_name}': {e}")
            return False

    def _is_model_ready_http(self, model_name: str, http_port: int) -> bool:
        """Check if model is ready using HTTP API."""
        try:
            response = self._http_request("GET", f"v2/models/{model_name}/ready", http_port, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _find_existing_container(self):
        """Finds if the shared container is already running on startup."""
        try:
            container_name = self.config["container_name"]
            result = self._run_command([
                "docker", "ps", "--filter", f"name={container_name}",
                "--format", "{{.ID}}\t{{.Status}}"
            ])
            if result.stdout.strip():
                container_info = result.stdout.strip().split('\t')
                container_id, status = container_info[0], container_info[1]
                if "Up" in status:
                    self.shared_container_id = container_id
                    self._on_server_started(self.config['http_port'])
                    print(f"Found existing running shared Triton container '{container_name}' (ID: {self.shared_container_id}).")
                else:
                    print(f"Found shared Triton container '{container_name}' but it's not running (Status: {status}).")
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"Could not check for existing shared container: {e}")

    def _start_server(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Starts a Triton Docker container."""
        if self.use_dedicated_server:
            http_port = find_free_port()
            grpc_port = find_free_port()
            metrics_port = find_free_port()
            container_name = f"triton_server_{model_name.lower()}_{uuid.uuid4().hex[:8]}"
            print(f"Starting dedicated Triton server for '{model_name}' on port {http_port}...")
        else:
            if self.shared_container_id:
                return {"container_id": self.shared_container_id, "http_port": self.config["http_port"]}
            http_port = self.config["http_port"]
            grpc_port = self.config["grpc_port"]
            metrics_port = self.config["metrics_port"]
            container_name = self.config["container_name"]
            print("Starting shared Triton Inference Server container...")

        cmd = [
            "docker", "run", "-d", "--rm", f"--shm-size={self.config['shm_size']}",
            "--name", container_name,
            "-v", f"{self.config['model_repository_host_path']}:/models",
            "-v", f"{self.config['runtime_host_path']}:/runtime",
            "-p", f"{http_port}:{http_port}",
            "-p", f"{grpc_port}:{grpc_port}",
            "-p", f"{metrics_port}:{metrics_port}",
        ]
        if self.config.get("gpus"):
            cmd.extend(["--gpus", self.config["gpus"]])
        
        cmd.append(self.config["image_name"])
        cmd.extend([
            "tritonserver",
            "--model-repository=/models",
            "--model-control-mode=explicit",
            f"--http-port={http_port}",
            f"--grpc-port={grpc_port}",
            f"--metrics-port={metrics_port}"
        ])

        result = self._run_command(cmd)
        container_id = result.stdout.strip()
        print(f"Container '{container_name}' started with ID: {container_id}")

        self._wait_for_server_ready(http_port)
        self._on_server_started(http_port)

        server_info = {"container_id": container_id, "http_port": http_port, "grpc_port": grpc_port, "metrics_port": metrics_port}
        if self.use_dedicated_server:
            self.dedicated_containers[model_name] = server_info
        else:
            self.shared_container_id = container_id
        
        return server_info

    def _on_server_started(self, http_port: int):
        """Actions to perform once a server container is confirmed running."""
        if self.use_dedicated_server:
            # No shared client in dedicated mode
            pass
        else:
            self.shared_triton_client = InferenceServerClient(url=f"localhost:{http_port}", verbose=False)
            self.shared_server_is_ready = True
            print(f"Shared Triton server on port {http_port} is ready.")
            self._sync_existing_models()

    def _sync_existing_models(self):
        """Synchronize internal state with models currently loaded in the shared Triton server."""
        if self.use_dedicated_server or not self.shared_server_is_ready:
            return
        
        try:
            http_port = self.config["http_port"]
            response = self._http_request("POST", "v2/repository/index", http_port, timeout=10)
            if response.status_code != 200:
                print(f"Failed to get repository index from shared server: {response.status_code}")
                return
                
            repository_data = response.json()
            
            for model_info in repository_data:
                model_name = model_info['name']
                model_state = model_info.get('state', 'UNKNOWN')
                
                if model_state == 'READY' and self._is_model_ready_http(model_name, http_port):
                    print(f"Syncing existing model '{model_name}' state from shared server...")
                    
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

    def _get_container_pid(self, container_id: str) -> Optional[int]:
        """Get the PID of a container."""
        if not container_id:
            return None
        try:
            result = self._run_command(["docker", "inspect", container_id, "--format", "{{.State.Pid}}"])
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None

    def _wait_for_server_ready(self, http_port: int):
        """Waits until a Triton server is live and ready."""
        print(f"Waiting for Triton server on port {http_port} to be ready...")
        timeout = self.config.get("api_timeout_seconds", 120)
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{http_port}/v2/health/live", timeout=5)
                if response.status_code == 200:
                    print(f"Triton server on port {http_port} is live.")
                    return
            except requests.exceptions.RequestException:
                pass  # Ignore connection errors while waiting
            time.sleep(1)
        raise TimeoutError(f"Triton server on port {http_port} failed to become ready in time.")

    def _get_instances_per_gpu(self, model_name: str) -> int:
        """Get the configured instances per GPU for a model."""
        model_configs = self.config.get("model_gpu_configs", {})
        if model_name in model_configs:
            return model_configs[model_name]["instances_per_gpu"]
        return model_configs.get("default", {}).get("instances_per_gpu", 1)

    def _update_model_config_and_reload(self, model_name: str, http_port: int):
        """Regenerate config and reload the model in a Triton server."""
        gpu_ids = list(self.model_gpu_mapping.get(model_name, []))
        
        if not gpu_ids:
            # Unload model if no GPUs are assigned
            try:
                if self._is_model_ready_http(model_name, http_port):
                    print(f"Unloading model '{model_name}' from server on port {http_port}...")
                    self._unload_model_http(model_name, http_port)
            except Exception as e:
                print(f"Error unloading model '{model_name}': {e}")
            remove_triton_config(model_name, self.config["model_repository_host_path"])
            return

        instances_per_gpu = self._get_instances_per_gpu(model_name)
        config_generated = generate_triton_config(
            model_name=model_name,
            model_repository_path=self.config["model_repository_host_path"],
            gpu_ids=gpu_ids,
            instance_count_per_gpu=instances_per_gpu
        )
        if not config_generated:
            raise FileNotFoundError(f"Could not generate config for {model_name}. Check for a template.")

        try:
            print(f"Initiating load/reload of model '{model_name}' on port {http_port}...")
            success = self._load_model_http(model_name, http_port)
            if not success:
                raise RuntimeError(f"Failed to initiate loading of model '{model_name}'")
            print(f"Model '{model_name}' load request sent successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to reload model '{model_name}': {e}")

    def _update_model_status(self, model_name: str):
        """Update model status from STARTING to RUNNING if model is ready."""
        if model_name in self.model_info and self.model_info[model_name].status == ModelInstanceStatus.STARTING:
            http_port = self.model_info[model_name].port
            if self._is_model_ready_http(model_name, http_port):
                self.model_info[model_name].status = ModelInstanceStatus.RUNNING
                self.model_info[model_name].updated_at = datetime.utcnow().isoformat()

    def shutdown(self):
        """Stops all managed Triton containers."""
        with self._lock:
            print("Shutting down all Triton containers managed by this runner...")
            
            # Shutdown shared container
            if self.shared_container_id:
                print(f"Stopping shared container {self.shared_container_id}...")
                try:
                    self._run_command(["docker", "stop", self.shared_container_id])
                    print(f"Shared container {self.shared_container_id} stopped.")
                except subprocess.CalledProcessError as e:
                    print(f"Error stopping shared container {self.shared_container_id}: {e}")
                finally:
                    self.shared_container_id = None
            
            # Shutdown all dedicated containers
            for model_name, container_info in list(self.dedicated_containers.items()):
                container_id = container_info["container_id"]
                print(f"Stopping dedicated container {container_id} for model '{model_name}'...")
                try:
                    self._run_command(["docker", "stop", container_id])
                    print(f"Dedicated container {container_id} stopped.")
                except subprocess.CalledProcessError as e:
                    print(f"Error stopping dedicated container {container_id}: {e}")
            
            self.dedicated_containers.clear()
            print("All Triton containers have been shut down.")

    def start(self, **kwargs) -> ModelInstanceInfo:
        """Add a GPU to a model's instance group, starting a server if needed."""
        model_name = kwargs["model_name"]
        gpu_id = kwargs["gpu_id"]

        with self._lock:
            # Determine which server to use or start
            if self.use_dedicated_server:
                if model_name not in self.dedicated_containers:
                    server_info = self._start_server(model_name)
                else:
                    server_info = self.dedicated_containers[model_name]
            else:
                server_info = self._start_server()

            container_id = server_info["container_id"]
            http_port = server_info["http_port"]

            # Initialize model tracking
            if model_name not in self.model_gpu_mapping:
                self.model_gpu_mapping[model_name] = set()
                self.model_info[model_name] = ModelInstanceInfo(
                    model_name=model_name, model_type="triton", active_gpu_ids=[],
                    total_instances=0, status=ModelInstanceStatus.STARTING,
                    pid=self._get_container_pid(container_id), port=http_port,
                    created_at=datetime.utcnow().isoformat(), updated_at=datetime.utcnow().isoformat()
                )

            if gpu_id in self.model_gpu_mapping[model_name]:
                print(f"GPU {gpu_id} is already assigned to model '{model_name}'")
                return self.model_info[model_name]

            original_gpus = self.model_gpu_mapping[model_name].copy()
            self.model_gpu_mapping[model_name].add(gpu_id)

            try:
                self._update_model_config_and_reload(model_name, http_port)
                
                instances_per_gpu = self._get_instances_per_gpu(model_name)
                self.model_info[model_name].active_gpu_ids = sorted(list(self.model_gpu_mapping[model_name]))
                self.model_info[model_name].total_instances = len(self.model_gpu_mapping[model_name]) * instances_per_gpu
                self.model_info[model_name].status = ModelInstanceStatus.STARTING
                self.model_info[model_name].updated_at = datetime.utcnow().isoformat()

                print(f"Successfully initiated start of model '{model_name}' on GPU {gpu_id}")
                return self.model_info[model_name]

            except Exception as e:
                self.model_gpu_mapping[model_name] = original_gpus
                if not original_gpus:
                    del self.model_gpu_mapping[model_name]
                    del self.model_info[model_name]
                    if self.use_dedicated_server and model_name in self.dedicated_containers:
                        self._stop_dedicated_server(model_name)
                
                raise e

    def _stop_dedicated_server(self, model_name: str):
        """Stops and removes a dedicated server container."""
        if model_name in self.dedicated_containers:
            container_id = self.dedicated_containers[model_name]["container_id"]
            print(f"Stopping dedicated container {container_id} for model '{model_name}'...")
            try:
                self._run_command(["docker", "stop", container_id])
                print(f"Dedicated container {container_id} stopped.")
            except subprocess.CalledProcessError as e:
                print(f"Error stopping dedicated container {container_id}: {e}")
            finally:
                del self.dedicated_containers[model_name]

    def stop(self, **kwargs) -> Optional[ModelInstanceInfo]:
        """Remove a GPU from a model's instance group."""
        model_name = kwargs["model_name"]
        gpu_id = kwargs["gpu_id"]

        with self._lock:
            if model_name not in self.model_gpu_mapping:
                print(f"Model '{model_name}' is not currently running.")
                return None

            if gpu_id not in self.model_gpu_mapping[model_name]:
                print(f"GPU {gpu_id} is not assigned to model '{model_name}'")
                return self.model_info[model_name]

            self.model_gpu_mapping[model_name].discard(gpu_id)
            
            http_port = self.model_info[model_name].port

            try:
                self._update_model_config_and_reload(model_name, http_port)
                
                if not self.model_gpu_mapping[model_name]:
                    # No GPUs left, stop the model and potentially the server
                    del self.model_gpu_mapping[model_name]
                    final_info = self.model_info.pop(model_name)
                    final_info.status = ModelInstanceStatus.STOPPED
                    final_info.active_gpu_ids = []
                    final_info.total_instances = 0
                    final_info.updated_at = datetime.utcnow().isoformat()
                    
                    print(f"Model '{model_name}' completely stopped.")
                    if self.use_dedicated_server:
                        self._stop_dedicated_server(model_name)
                    
                    return final_info
                else:
                    # Update model info with remaining GPUs
                    instances_per_gpu = self._get_instances_per_gpu(model_name)
                    self.model_info[model_name].active_gpu_ids = sorted(list(self.model_gpu_mapping[model_name]))
                    self.model_info[model_name].total_instances = len(self.model_gpu_mapping[model_name]) * instances_per_gpu
                    self.model_info[model_name].updated_at = datetime.utcnow().isoformat()
                    
                    print(f"Removed GPU {gpu_id} from model '{model_name}'.")
                    return self.model_info[model_name]
                    
            except Exception as e:
                print(f"Error stopping model '{model_name}' on GPU {gpu_id}: {e}")
                # Restore GPU to mapping on failure
                self.model_gpu_mapping[model_name].add(gpu_id)
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