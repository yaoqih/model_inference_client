"""
Pydantic schemas for API data validation.
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """
    Enum for the supported model serving backends.
    """
    TRITON = "triton"
    COMFYUI = "comfyui"


class ModelInstanceStatus(str, Enum):
    """
    Enum for the status of a model instance.
    """
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class StartRequest(BaseModel):
    """
    Request schema for starting a model on a specific GPU.
    """
    model_name: str = Field(..., description="The name of the model to start.")
    model_type: Optional[ModelType] = Field(None, description="The type of the serving backend. If not provided, will be auto-detected from model name.")
    gpu_id: int = Field(..., ge=0, description="GPU ID to add instance on.")
    config: Dict[str, Any] = Field({}, description="Backend-specific configuration options.")


class StopRequest(BaseModel):
    """
    Request schema for stopping a model on a specific GPU.
    """
    model_name: str = Field(..., description="The name of the model.")
    model_type: Optional[ModelType] = Field(None, description="The type of the serving backend. If not provided, will be auto-detected from model name.")
    gpu_id: int = Field(..., ge=0, description="GPU ID to remove instance from.")
    instance_id: Optional[str] = Field(None, description="The specific instance ID to stop, if applicable.")


class ModelInstanceInfo(BaseModel):
    """
    Response schema representing a model's status.
    """
    instance_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this model instance.")
    model_name: str = Field(..., description="Name of the model.")
    model_type: ModelType = Field(..., description="Type of the serving backend.")
    active_gpu_ids: List[int] = Field(..., description="List of GPUs currently running this model.")
    total_instances: int = Field(..., description="Total number of instances across all GPUs.")
    status: ModelInstanceStatus = Field(..., description="Current status of the model.")
    pid: Optional[int] = Field(None, description="Process ID of the server (for container-based backends).")
    port: Optional[int] = Field(None, description="Port the model server is listening on.")
    created_at: str = Field(..., description="Timestamp of when the model was first started.")
    updated_at: str = Field(..., description="Timestamp of when the model status was last updated.")

    class Config:
        use_enum_values = True


class ModelStatusItem(BaseModel):
    """
    Simplified model status response item for each GPU instance.
    """
    model_name: str = Field(..., description="Name of the model.")
    model_type: str = Field(..., description="Type of the serving backend.")
    gpu_id: int = Field(..., description="GPU ID running this model instance.")

    class Config:
        use_enum_values = True

from pydantic import BaseModel
from typing import List, Optional

class GPUProcess(BaseModel):
    pid: int
    name: Optional[str] = 'N/A'
    gpu_memory_usage: int

class GPUInfo(BaseModel):
    id: int
    memory_usage: float
    power_draw: float
    processes: List[GPUProcess]

class KillProcessResponse(BaseModel):
    pid: int
    status: str
    message: Optional[str] = None
