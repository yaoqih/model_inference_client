"""
Global configuration for the Model Inference Client.
"""
import os

# Base directory of the model_inference_client package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRITON_RUNNER_CONFIG = {
    "image_name": "nvcr.io/nvidia/tritonserver:25.04-py3",
    "container_name": "triton_server",
    "model_repository_host_path": '/mnt/sdb_data/triton_inference_server/model_repository',
    "runtime_host_path":'/mnt/sdb_data/triton_inference_server/runtime',
    "http_port": 8000,
    "grpc_port": 8001,
    "metrics_port": 8002,
    "shm_size": "2g",
    "gpus": "all",  # Use "all" to expose all GPUs, or specify device IDs like "device=0,1"
    "api_timeout_seconds": 120, # Timeout for waiting for Triton server to be ready
    "model_gpu_configs": {
        # Each model can have different instance counts per GPU
        "MAM": {
            "instances_per_gpu": 8,  # Default instances per GPU for MAM model
        },
        "FastFitAll": {
            "instances_per_gpu": 1,  # Default instances per GPU for FastFitAll model
        },
        # Add more models as needed
        "default": {
            "instances_per_gpu": 1,  # Default for models not explicitly configured
        }
    }
}

COMFYUI_RUNNER_CONFIG = {
    # TODO: Add ComfyUI specific configs here
}

# A central mapping of runner configurations
RUNNERS_CONFIG = {
    "triton": TRITON_RUNNER_CONFIG,
    "comfyui": COMFYUI_RUNNER_CONFIG,
}
