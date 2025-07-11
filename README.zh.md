# 模型推理客户端

一个用于启动、停止和监控如 Triton、ComfyUI 等模型推理服务的 API 服务器。该服务器提供了一个统一的接口，用于管理在不同 GPU 上运行的多个模型后端。

## 功能特性

-   **统一 API**: 通过单一 API 管理各种推理后端。
-   **多 GPU 支持**: 在多个 GPU 设备上运行和管理模型。
-   **动态模型管理**: 即时在特定 GPU 上启动和停止模型。
-   **后端无关**: 可轻松扩展以支持新的推理后端（例如 vLLM, Oobabooga）。
-   **GPU 监控**: 提供 API 端点以获取详细的 GPU 状态，包括显存使用情况、功耗和正在运行的进程。
-   **进程管理**: 提供一个通过 PID 终止进程的端点，用于清理僵尸进程。

## 安装

1.  **克隆仓库:**
    ```bash
    git clone <your-repo-url>
    cd model_inference_client
    ```

2.  **安装依赖:**
    建议使用虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## 使用方法

要启动 API 服务器，请在项目根目录运行以下命令：

```bash
uvicorn model_inference_client.main:app --host 0.0.0.0 --port 6004 --reload
```

服务器将在 `http://0.0.0.0:6004` 上可用。您可以通过 `http://0.0.0.0:6004/docs` 访问自动生成的 OpenAPI 文档。

## API 端点

所有端点都在 `/api/v1` 前缀下。

### GPU 和进程管理

| 方法     | 端点                  | 描述                                                                | 响应                               |
| :------- | :-------------------- | :------------------------------------------------------------------ | :--------------------------------- |
| `GET`    | `/gpus`               | 获取所有 GPU 的列表及其当前状态，包括正在运行的进程。                 | `List[GPUInfo]`                    |
| `DELETE` | `/processes/{pid}`    | 通过进程 ID (PID) 终止一个进程。**请谨慎使用。**                      | `KillProcessResponse`              |

#### `GPUInfo` 结构示例

```json
{
  "id": 0,
  "memory_usage": 45.5,
  "power_draw": 150.7,
  "processes": [
    {
      "pid": 12345,
      "name": "python",
      "gpu_memory_usage": 4096
    }
  ]
}
```

### 模型管理

| 方法   | 端点                        | 描述                                     | 请求体           | 响应                     |
| :----- | :-------------------------- | :--------------------------------------- | :--------------- | :----------------------- |
| `POST` | `/models/start`             | 在特定 GPU 上启动一个模型实例。          | `StartRequest`   | `ModelInstanceInfo`      |
| `POST` | `/models/stop`              | 在特定 GPU 上停止一个模型实例。          | `StopRequest`    | `ModelInstanceInfo`      |
| `GET`  | `/models/status`            | 获取所有正在运行的模型实例的状态。       | -                | `List[ModelStatusItem]`  |
| `GET`  | `/models/status/{model_name}` | 获取特定模型的所有实例的状态。           | -                | `List[ModelStatusItem]`  |

#### `StartRequest` 请求体示例

```json
{
  "model_name": "MAM",
  "model_type": "triton",
  "gpu_id": 0,
  "config": {}
}
```

## 配置

全局和特定运行器的配置位于 `model_inference_client/config.py`。您可以修改此文件以更改 Docker 镜像、端口、模型路径等设置。

## 扩展新的后端

该系统设计为易于扩展。要为新的模型服务后端（例如 vLLM）添加支持：

1.  **创建新的 Runner 类:**
    在 `model_inference_client/runners/` 中创建一个新文件，例如 `vllm_runner.py`。
    在此文件中，定义一个继承自 `BaseRunner` 的类。
    ```python
    from .base_runner import BaseRunner
    from ..models.schemas import ModelInstanceInfo

    class vLLMRunner(BaseRunner):
        def start(self, model_name: str, gpu_id: int, **kwargs) -> ModelInstanceInfo:
            # 添加启动 vLLM 服务器作为子进程的逻辑
            pass

        def stop(self, instance: ModelInstanceInfo) -> None:
            # 添加停止 vLLM 进程的逻辑
            pass

        def get_status(self, instance_id: str) -> ModelInstanceInfo:
            # 添加检查 vLLM 进程状态的逻辑
            pass
    ```

2.  **添加配置:**
    在 `model_inference_client/config.py` 中为您的运行器添加一个新的配置字典。

3.  **注册 Runner:**
    在 `model_inference_client/services/model_manager.py` 中，导入您的新运行器，并将其添加到 `_init_runners` 方法中的 `self.runners` 字典中。 