"""
Core service for managing the lifecycle of model instances.
"""
import datetime
import uuid
from threading import Lock
from typing import Dict, List, Optional, Type

from models.schemas import (ModelInstanceInfo,
                                                   ModelInstanceStatus,
                                                   ModelType, ModelStatusItem, StartRequest, StopRequest)
from runners.base_runner import BaseRunner
from runners.comfyui_runner import ComfyUIRunner
from runners.triton_runner import TritonRunner
from config import RUNNERS_CONFIG, MODEL_BACKEND_MAPPING


class ModelManager:
    """
    A thread-safe singleton manager for model instances.

    This class is responsible for creating, tracking, and managing the lifecycle
    of all model serving processes.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict = None):
        """
        Initializes the ModelManager.
        """
        # Ensure __init__ is called only once for the singleton
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    self.config = config or {}
                    self._init_runners(RUNNERS_CONFIG)
                    self.initialized = True

    def _init_runners(self, runner_configs: Dict):
        """Initializes the available runners from the global config."""
        self.runners: Dict[ModelType, BaseRunner] = {
            ModelType.TRITON: TritonRunner(runner_configs.get("triton", {})),
            ModelType.COMFYUI: ComfyUIRunner(runner_configs.get("comfyui", {}))
        }
        print("ModelManager initialized with runners.")

    def _detect_model_backend(self, model_name: str) -> ModelType:
        """
        根据模型名称自动检测推理后端类型
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelType: 推理后端类型
            
        Raises:
            ValueError: 如果模型名称不在支持列表中
        """
        if model_name in MODEL_BACKEND_MAPPING:
            backend_str = MODEL_BACKEND_MAPPING[model_name]
            return ModelType(backend_str)
        
        # 收集所有支持的模型名称用于错误信息
        all_supported_models = set(MODEL_BACKEND_MAPPING.keys())
        
        # 如果都没找到，抛出异常
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported models: {sorted(list(all_supported_models))}"
        )

    def start_model(self, request: StartRequest) -> ModelInstanceInfo:
        """
        启动模型，支持自动检测后端类型
        
        Args:
            request: The StartRequest object from the API.

        Returns:
            Information about the model after starting.
        
        Raises:
            ValueError: If the requested model_type is not supported.
        """
        # 如果没有指定model_type，自动检测
        if request.model_type is None:
            request.model_type = self._detect_model_backend(request.model_name)
        
        # 验证model_type和model_name的组合是否有效
        runner = self.runners.get(request.model_type)
        if not runner:
            raise ValueError(f"Unsupported model type: {request.model_type}")
        
        # 验证模型是否在该runner的支持列表中
        if hasattr(runner, 'config') and 'supported_models' in runner.config:
            if request.model_name not in runner.config['supported_models']:
                raise ValueError(f"Model '{request.model_name}' is not supported by {request.model_type} backend")

        try:
            model_info = runner.start(**request.dict())
            print(f"Model '{request.model_name}' started on GPU {request.gpu_id} using {request.model_type} backend")
            return model_info
        except Exception as e:
            print(f"Failed to start model '{request.model_name}': {e}")
            raise e

    def stop_model(self, request: StopRequest) -> Optional[ModelInstanceInfo]:
        """
        停止模型，支持自动检测后端类型
        
        Args:
            request: The StopRequest object from the API.

        Returns:
            The final status information of the model, or None if not found.
        """
        # 如果没有指定model_type，自动检测
        if request.model_type is None:
            request.model_type = self._detect_model_backend(request.model_name)
        
        runner = self.runners.get(request.model_type)
        if not runner:
            raise ValueError(f"Unsupported model type: {request.model_type}")

        try:
            model_info = runner.stop(**request.dict())
            if model_info:
                print(f"Model '{request.model_name}' stopped on GPU {request.gpu_id}")
            return model_info
        except Exception as e:
            print(f"Failed to stop model '{request.model_name}': {e}")
            raise e

    def get_model_status(self, model_name: str, model_type: ModelType) -> Optional[ModelInstanceInfo]:
        """
        Gets the status of a specific model.

        Args:
            model_name: The name of the model to query.
            model_type: The type of the model backend.

        Returns:
            The status information, or None if not found.
        """
        runner = self.runners.get(model_type)
        if not runner:
            return None
        
        return runner.get_status(model_name)

    def get_model_status_by_name(self, model_name: str) -> Optional[ModelInstanceInfo]:
        """
        Gets the status of a specific model by searching all runners.

        Args:
            model_name: The name of the model to query.

        Returns:
            The status information, or None if not found.
        """
        for model_type, runner in self.runners.items():
            if hasattr(runner, 'get_status'):
                model_info = runner.get_status(model_name)
                if model_info:
                    return model_info
        
        return None

    def get_all_statuses(self) -> List[ModelInstanceInfo]:
        """
        Gets the status of all managed models.

        Returns:
            A list of status information for all models.
        """
        all_models = []
        
        for model_type, runner in self.runners.items():
            if hasattr(runner, 'get_all_active_models'):
                all_models.extend(runner.get_all_active_models())
            
        return all_models

    def get_all_statuses_simplified(self) -> List[ModelStatusItem]:
        """
        Gets the simplified status of all managed models.
        Returns one item per GPU instance.

        Returns:
            A list of simplified status items for all model instances.
        """
        simplified_items = []
        
        for model_type, runner in self.runners.items():
            if hasattr(runner, 'get_all_active_models'):
                models = runner.get_all_active_models()
                for model in models:
                    # Create one item for each GPU the model is running on
                    for gpu_id in model.active_gpu_ids:
                        simplified_items.append(ModelStatusItem(
                            model_name=model.model_name,
                            model_type=model.model_type,
                            gpu_id=gpu_id
                        ))
        
        return simplified_items

    def get_model_status_by_name_simplified(self, model_name: str) -> List[ModelStatusItem]:
        """
        Gets the simplified status of a specific model by searching all runners.
        Returns one item per GPU instance.

        Args:
            model_name: The name of the model to query.

        Returns:
            A list of simplified status items for the model, or empty list if not found.
        """
        simplified_items = []
        
        for model_type, runner in self.runners.items():
            if hasattr(runner, 'get_status'):
                model_info = runner.get_status(model_name)
                if model_info:
                    # Create one item for each GPU the model is running on
                    for gpu_id in model_info.active_gpu_ids:
                        simplified_items.append(ModelStatusItem(
                            model_name=model_info.model_name,
                            model_type=model_info.model_type,
                            gpu_id=gpu_id
                        ))
                    break  # Found the model, no need to check other runners
        
        return simplified_items

    def get_supported_models(self) -> Dict[str, str]:
        """
        获取支持的模型列表及其对应的后端类型
        
        Returns:
            Dict[str, str]: 模型名称到后端类型的映射
        """
        supported_models = {}
        
        # 首先从全局映射中获取
        supported_models.update(MODEL_BACKEND_MAPPING)  
        
        return supported_models

    def shutdown(self):
        """
        Stops all running models and shuts down runners.
        This is intended to be called on application shutdown.
        """
        print("Shutting down all model runners...")
        
        for model_type, runner in self.runners.items():
            if hasattr(runner, 'shutdown'):
                try:
                    runner.shutdown()
                except Exception as e:
                    print(f"Error shutting down {model_type} runner: {e}")

        print("All model runners have been shut down.")


# Create a single, globally accessible instance of the manager
model_manager = ModelManager(config={}) # Pass empty config, as it now reads from config.py
