# 模型推理客户端 (Model Inference Client)

一个用于启动、停止和监控如 Triton、ComfyUI 等模型推理服务的统一 API 服务器。面向多模型、多后端场景，提供一致的控制与可观测能力，并内置 GPU 独占与并发串行化机制。

相关文档与源码索引：
- [docs/handover.md](docs/handover.md) 交接文档（面向接手工程师，强烈建议先阅读）
- [pyproject.toml](pyproject.toml)
- [src/model_inference_client/main.py](src/model_inference_client/main.py)
- [src/model_inference_client/api/routes.py](src/model_inference_client/api/routes.py)
- [src/model_inference_client/services/model_manager.py](src/model_inference_client/services/model_manager.py)
- [src/model_inference_client/runners/triton_runner.py](src/model_inference_client/runners/triton_runner.py)
- [src/model_inference_client/runners/comfyui_runner.py](src/model_inference_client/runners/comfyui_runner.py)
- [src/model_inference_client/models/schemas.py](src/model_inference_client/models/schemas.py)
- [src/model_inference_client/utils/gpu_utils.py](src/model_inference_client/utils/gpu_utils.py)
- [src/model_inference_client/utils/network_utils.py](src/model_inference_client/utils/network_utils.py)
- [src/model_inference_client/utils/config_utils.py](src/model_inference_client/utils/config_utils.py)
- [src/model_inference_client/config.py](src/model_inference_client/config.py)

## 功能特性

- 统一 API：通过单一 API 管理多种推理后端（Triton、ComfyUI）。
- 多 GPU 支持：精确控制模型实例运行在哪张 GPU 上。
- 动态模型管理：在特定 GPU 上即时启动/停止实例。
- 后端无关与可扩展：按 Runner 插件化扩展新的后端（如 vLLM）。
- GPU 监控：提供详细 GPU 状态（显存、功耗、进程）。
- 进程管理：可按 PID 终止进程清理僵尸子进程。
- GPU 独占：同一时刻每张 GPU 仅允许一个模型实例（防止资源冲突）。
- 异步与串行化：服务端与 Runner 操作采用 asyncio + Lock 保证原子性与一致性。

## 目录

- 快速开始
- API 概览与示例
- 配置说明
- 架构概览
- GPU 独占与并发控制
- 扩展新的后端 Runner
- 故障排查与常见问题
- 许可证

## 快速开始

环境准备
- Python 3.12+
- Docker（运行 Triton 时必需）
- NVIDIA 驱动与 CUDA（可选，用于 GPU 监控与实际推理）
- 依赖见 [pyproject.toml](pyproject.toml)

安装依赖（推荐使用 uv）
- 安装 uv：`pip install uv`
- 安装项目依赖（使用 TUNA 镜像已配置于 pyproject）：  
  `uv pip install -r <(python - <<'PY'\nimport tomllib,sys\ncfg=tomllib.loads(open('pyproject.toml','rb').read())\nprint('\\n'.join(cfg['project']['dependencies']))\nPY\n)`

或使用 venv + pip
- 创建虚拟环境：
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- 安装依赖（逐条 pip install，参照 [pyproject.toml](pyproject.toml) 中的 [project.dependencies]）

启动 API 服务器（源码方式，推荐）
```bash
python -m uvicorn src.model_inference_client.main:app --host 0.0.0.0 --port 6004 --reload
```
- 访问 API 根路由健康检查：`http://0.0.0.0:6004/`  
  对应处理函数： [python.read_root()](src/model_inference_client/main.py:35)
- 访问 OpenAPI 文档：`http://0.0.0.0:6004/docs`

可选：按包入口启动（需先构建/安装）
- 入口函数： [python.run()](src/model_inference_client/main.py:42)
- pyproject 脚本入口：[pyproject.toml](pyproject.toml)

注意：如果使用 `uvicorn model_inference_client.main:app`，需确保 Python 路径包含 `src`（例如设置 `PYTHONPATH=src`）或已将包安装进环境。

## API 概览与示例

所有端点在前缀 `/api/v1` 下，定义于 [src/model_inference_client/api/routes.py](src/model_inference_client/api/routes.py)。

- GPU 与进程
  - GET `/gpus`（查看 GPU 列表与状态）  
    处理函数： [python.get_gpus_status()](src/model_inference_client/api/routes.py:22)  
    结构： [python.GPUInfo()](src/model_inference_client/models/schemas.py:88)
  - DELETE `/processes/{pid}`（按 PID 杀进程）  
    处理函数： [python.kill_process()](src/model_inference_client/api/routes.py:42)

- 模型管理
  - POST `/models/start`（在指定 GPU 启动模型实例）  
    处理函数： [python.start_model_on_gpu()](src/model_inference_client/api/routes.py:64)  
    请求体模型： [python.StartRequest()](src/model_inference_client/models/schemas.py:30)  
    响应模型： [python.ModelInstanceInfo()](src/model_inference_client/models/schemas.py:50)
  - POST `/models/stop`（停止指定 GPU 上的模型实例）  
    处理函数： [python.stop_model_on_gpu()](src/model_inference_client/api/routes.py:88)  
    请求体模型： [python.StopRequest()](src/model_inference_client/models/schemas.py:40)  
    响应模型： [python.ModelInstanceInfo()](src/model_inference_client/models/schemas.py:50) 或空
  - GET `/models/status`（获取所有运行中实例的简化状态）  
    处理函数： [python.get_all_models_status()](src/model_inference_client/api/routes.py:117)  
    响应模型： [python.ModelStatusItem()](src/model_inference_client/models/schemas.py:69) 列表
  - GET `/models/status/{model_name}`（获取某模型的所有实例简化状态）  
    处理函数： [python.get_model_status()](src/model_inference_client/api/routes.py:130)
  - GET `/models/supported`（可用模型与后端映射）  
    处理函数： [python.get_supported_models()](src/model_inference_client/api/routes.py:149)

示例请求
- 启动模型（后端类型可省略，由映射自动推断）：
  ```bash
  curl -X POST http://localhost:6004/api/v1/models/start \
    -H "Content-Type: application/json" \
    -d '{"model_name":"MAM","gpu_id":0,"config":{}}'
  ```
  StartRequest 字段： [python.StartRequest()](src/model_inference_client/models/schemas.py:30)

- 停止模型：
  ```bash
  curl -X POST http://localhost:6004/api/v1/models/stop \
    -H "Content-Type: application/json" \
    -d '{"model_name":"MAM","gpu_id":0}'
  ```

- 查看状态：
  ```bash
  curl http://localhost:6004/api/v1/models/status
  ```

## 配置说明

集中配置文件： [src/model_inference_client/config.py](src/model_inference_client/config.py)

关键项
- 模型到后端自动映射： [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11)  
  例如：`"MAM": "triton"`, `"chane_face": "comfyui"`
- Triton 运行器： [python.TRITON_RUNNER_CONFIG](src/model_inference_client/config.py:17)
  - `use_dedicated_server`：True=每模型独立容器（端口动态），False=共享容器（端口固定）
  - `image_name`、`container_name`：Docker 镜像与容器名
  - `model_repository_host_path`：模型仓库目录（需包含每模型的 `config.pbtxt.template`）
  - `runtime_host_path`：运行时挂载目录
  - `http_port`/`grpc_port`/`metrics_port`：端口（共享模式）
  - `gpus`：暴露设备策略，如 "all" 或 "device=0,1"
  - `api_timeout_seconds`：Triton 就绪等待超时
  - `model_gpu_configs`：每模型 `instances_per_gpu` 配置
- ComfyUI 运行器：在 `COMFYUI_RUNNER_CONFIG` 中设置 `comfyui_path`（指向包含 `main.py` 的 ComfyUI 目录）

辅助工具
- 端口分配： [python.find_free_port()](src/model_inference_client/utils/network_utils.py:6)
- 生成 Triton 配置： [python.generate_triton_config()](src/model_inference_client/utils/config_utils.py:9) 与 [python.remove_triton_config()](src/model_inference_client/utils/config_utils.py:63)
- GPU 信息： [python.get_gpu_info()](src/model_inference_client/utils/gpu_utils.py:25)

## 架构概览

调用路径（核心链路）
- 启动：API 路由 → [python.ModelManager.start_model()](src/model_inference_client/services/model_manager.py:89) → Runner.start（如 [python.TritonRunner.start()](src/model_inference_client/runners/triton_runner.py:409)、[python.ComfyUIRunner.start()](src/model_inference_client/runners/comfyui_runner.py:47)）
- 停止：API 路由 → [python.ModelManager.stop_model()](src/model_inference_client/services/model_manager.py:139) → Runner.stop
- 状态：API 路由 → [python.ModelManager.get_all_statuses_simplified()](src/model_inference_client/services/model_manager.py:228) → Runner.get_all_active_models

组件职责
- FastAPI 应用与路由： [src/model_inference_client/main.py](src/model_inference_client/main.py) · [src/model_inference_client/api/routes.py](src/model_inference_client/api/routes.py)
- 服务层单例： [src/model_inference_client/services/model_manager.py](src/model_inference_client/services/model_manager.py)
- Runner 层： [src/model_inference_client/runners/triton_runner.py](src/model_inference_client/runners/triton_runner.py) · [src/model_inference_client/runners/comfyui_runner.py](src/model_inference_client/runners/comfyui_runner.py)
- 配置： [src/model_inference_client/config.py](src/model_inference_client/config.py)
- 数据模型（Pydantic）： [src/model_inference_client/models/schemas.py](src/model_inference_client/models/schemas.py)
- 工具： [src/model_inference_client/utils](src/model_inference_client/utils)

## GPU 独占与并发控制

- 每张 GPU 有独立的 `asyncio.Lock`，在 [python.ModelManager.__init__()](src/model_inference_client/services/model_manager.py:37) 中初始化并维护 `_active_gpu_models` 状态，确保任一时刻每张 GPU 仅运行一个模型。
- 若 GPU 已被占用，启动请求将返回错误（见 [python.ModelManager.start_model()](src/model_inference_client/services/model_manager.py:106)）。
- 各 Runner 内部亦使用 `asyncio.Lock` 串行化执行（如 [python.TritonRunner.__init__()](src/model_inference_client/runners/triton_runner.py:30)、[python.ComfyUIRunner.__init__()](src/model_inference_client/runners/comfyui_runner.py:24)），保证 start/stop 原子性。

## 扩展新的后端 Runner

步骤
1) 在 [src/model_inference_client/runners](src/model_inference_client/runners) 新增文件并实现异步接口（参考 [python.TritonRunner.start()](src/model_inference_client/runners/triton_runner.py:409)、[python.ComfyUIRunner.start()](src/model_inference_client/runners/comfyui_runner.py:47)）。
2) 在 [python.ModelManager._init_runners()](src/model_inference_client/services/model_manager.py:55) 中注册新 Runner。
3) 在 [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11) 中维护模型名称 → 后端类型映射。
4) 保持响应模型与状态语义一致： [python.ModelInstanceInfo()](src/model_inference_client/models/schemas.py:50) / [python.ModelStatusItem()](src/model_inference_client/models/schemas.py:69)。

注意
- 使用 `asyncio.Lock` 保证 Runner 内操作串行化。
- 长耗时操作（容器启动、HTTP 加载模型）必须异步实现，避免阻塞事件循环。

## 故障排查与常见问题

- NVML 未安装或无 GPU：`/gpus` 可能返回 501 或设备数为 0；请安装 `nvidia-ml-py` 与 NVIDIA 驱动。
- Docker 不可用：Triton 容器无法启动；检查 `docker ps` 可用性与权限。
- 缺少 `config.pbtxt.template`：Triton 模型无法加载；请在模型目录补齐模板文件。
- 端口占用：共享模式使用固定端口；独立模式端口由 [python.find_free_port()](src/model_inference_client/utils/network_utils.py:6) 动态选择，仍需关注安全策略与防火墙。
- GPU 已被占用：同一 GPU 同时仅允许一个模型；请选择空闲 GPU 或先停止已占用模型。

## 许可证

---

更完整的运维、并发机制与 Runner 细节，请阅读交接文档：[docs/handover.md](docs/handover.md)。