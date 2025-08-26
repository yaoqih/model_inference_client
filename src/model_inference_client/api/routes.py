"""
API routes for the model inference client.
"""
from typing import List, Dict

from fastapi import APIRouter, HTTPException, Body, Query
from starlette import status

from model_inference_client.models.schemas import (ModelInstanceInfo, ModelType, ModelStatusItem,
                                                   StartRequest, StopRequest, GPUInfo, KillProcessResponse)
from model_inference_client.services.model_manager import model_manager
from model_inference_client.utils.gpu_utils import get_gpu_info, kill_process_by_pid

router = APIRouter()


@router.get(
    "/gpus",
    response_model=List[GPUInfo],
    summary="Get status of all GPUs"
)
async def get_gpus_status():
    """
    Retrieves the status of all GPUs on this machine.
    """
    try:
        return get_gpu_info()
    except ImportError as e:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU info: {e}",
        )


@router.delete(
    "/processes/{pid}",
    response_model=KillProcessResponse,
    summary="Kill a process by its PID"
)
async def kill_process(pid: int):
    """
    Kills a process on the machine by its Process ID.
    """
    result = kill_process_by_pid(pid)
    if result["status"] == "error":
        if "not found" in result["message"]:
            status_code = status.HTTP_404_NOT_FOUND
        elif "Permission" in result["message"]:
            status_code = status.HTTP_403_FORBIDDEN
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(status_code=status_code, detail=result["message"])
    return result


@router.post(
    "/models/start",
    response_model=ModelInstanceInfo,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a model on a specific GPU"
)
async def start_model_on_gpu(request: StartRequest = Body(...)):
    """
    Starts a model on the specified GPU.
    The model backend type will be automatically detected from the model name if not specified.
    If the model is already running on other GPUs, the new GPU will be added.
    """
    try:
        model_info = await model_manager.start_model(request)
        return model_info
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # A catch-all for other unexpected errors during startup
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model: {e}",
        )


@router.post(
    "/models/stop",
    response_model=ModelInstanceInfo,
    summary="Stop a model on a specific GPU"
)
async def stop_model_on_gpu(request: StopRequest = Body(...)):
    """
    Stops a model on the specified GPU.
    The model backend type will be automatically detected from the model name if not specified.
    If the model is running on other GPUs, those will continue running.
    If this is the last GPU, the model will be completely stopped.
    """
    try:
        model_info = await model_manager.stop_model(request)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' not found or not running on GPU {request.gpu_id}.",
            )
        return model_info
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop model: {e}",
        )


@router.get(
    "/models/status",
    response_model=List[ModelStatusItem],
    summary="Get status of all running models"
)
async def get_all_models_status():
    """
    Retrieves the status of all managed models.
    Returns a list where each item represents a model instance on a specific GPU.
    """
    return await model_manager.get_all_statuses_simplified()


@router.get(
    "/models/status/{model_name}",
    response_model=List[ModelStatusItem],
    summary="Get status of a specific model"
)
async def get_model_status(model_name: str):
    """
    Retrieves the status of a single model by its name.
    Returns a list where each item represents a model instance on a specific GPU.
    """
    model_items = await model_manager.get_model_status_by_name_simplified(model_name)
    if not model_items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )
    return model_items


@router.get(
    "/models/supported",
    response_model=Dict[str, str],
    summary="Get list of supported models and their backends"
)
async def get_supported_models():
    """
    Returns a mapping of supported model names to their backend types.
    This endpoint helps users understand which models are available and
    which backend will be used for each model.
    """
    return model_manager.get_supported_models()
