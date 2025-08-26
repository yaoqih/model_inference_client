# 交接文档：Model Inference Client

本文面向将要接手本项目的工程师，涵盖架构、运行、配置、API、并发与GPU独占机制、Runner扩展、部署与运维、排障与常见问题、以及质量保障与发布流程。

关联文件： [README.md](README.md) · [pyproject.toml](pyproject.toml) · [src/model_inference_client/main.py](src/model_inference_client/main.py) · [src/model_inference_client/api/routes.py](src/model_inference_client/api/routes.py) · [src/model_inference_client/services/model_manager.py](src/model_inference_client/services/model_manager.py) · [src/model_inference_client/runners/triton_runner.py](src/model_inference_client/runners/triton_runner.py) · [src/model_inference_client/runners/comfyui_runner.py](src/model_inference_client/runners/comfyui_runner.py) · [src/model_inference_client/config.py](src/model_inference_client/config.py) · [src/model_inference_client/models/schemas.py](src/model_inference_client/models/schemas.py) · [src/model_inference_client/utils/gpu_utils.py](src/model_inference_client/utils/gpu_utils.py) · [src/model_inference_client/utils/network_utils.py](src/model_inference_client/utils/network_utils.py) · [src/model_inference_client/utils/config_utils.py](src/model_inference_client/utils/config_utils.py)

一、项目概览

- 作用：统一管理多种推理后端（Triton、ComfyUI），提供启动/停止/查询与GPU监控的REST API。
- 技术：FastAPI、异步运行器、Docker（Triton）、HTTP RPC（Triton V2 API）、NVML（GPU状态）。
- 运行形态：开发模式直接用 uvicorn 启动；生产可通过容器或进程管理器运行。

二、代码结构

- 应用入口： [python.lifespan()](src/model_inference_client/main.py:13) · [python.run()](src/model_inference_client/main.py:42) · [python.FastAPI()](src/model_inference_client/main.py:26)
- API 路由： [python.get_gpus_status()](src/model_inference_client/api/routes.py:22) · [python.kill_process()](src/model_inference_client/api/routes.py:42) · [python.start_model_on_gpu()](src/model_inference_client/api/routes.py:64) · [python.stop_model_on_gpu()](src/model_inference_client/api/routes.py:88) · [python.get_all_models_status()](src/model_inference_client/api/routes.py:117) · [python.get_model_status()](src/model_inference_client/api/routes.py:130) · [python.get_supported_models()](src/model_inference_client/api/routes.py:149)
- 服务层（单例）： [python.ModelManager.start_model()](src/model_inference_client/services/model_manager.py:89) · [python.ModelManager.stop_model()](src/model_inference_client/services/model_manager.py:139) · [python.ModelManager.get_all_statuses_simplified()](src/model_inference_client/services/model_manager.py:228)
- Runner： [python.TritonRunner.start()](src/model_inference_client/runners/triton_runner.py:409) · [python.TritonRunner.stop()](src/model_inference_client/runners/triton_runner.py:479) · [python.ComfyUIRunner.start()](src/model_inference_client/runners/comfyui_runner.py:47) · [python.ComfyUIRunner.stop()](src/model_inference_client/runners/comfyui_runner.py:101)
- 配置： [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11) · [python.TRITON_RUNNER_CONFIG](src/model_inference_client/config.py:17) · [python.RUNNERS_CONFIG](src/model_inference_client/config.py:49)
- 数据模型： [python.StartRequest()](src/model_inference_client/models/schemas.py:30) · [python.StopRequest()](src/model_inference_client/models/schemas.py:40) · [python.ModelInstanceInfo()](src/model_inference_client/models/schemas.py:50) · [python.ModelStatusItem()](src/model_inference_client/models/schemas.py:69) · [python.GPUInfo()](src/model_inference_client/models/schemas.py:88)
- 工具： [python.get_gpu_info()](src/model_inference_client/utils/gpu_utils.py:25) · [python.get_gpu_count()](src/model_inference_client/utils/gpu_utils.py:10) · [python.find_free_port()](src/model_inference_client/utils/network_utils.py:6) · [python.generate_triton_config()](src/model_inference_client/utils/config_utils.py:9)

三、运行与本地开发

先决条件
- Python 3.12+
- NVIDIA 驱动（可选，用于GPU监控端点）
- Docker（用于 Triton）
- 可选：nvml Python 包（由依赖 nvidia-ml-py 提供）

安装依赖（开发者推荐）
- 使用 uv（更快的包管理）：
  1) 安装 uv： pip install uv
  2) 安装依赖： uv pip install -r <(python -c "import tomllib,sys;print('\\n'.join(__import__('tomllib').loads(open('pyproject.toml','rb').read())['project']['dependencies']))")
- 或使用 pip（无构建安装）：
  1) 创建虚拟环境： python -m venv .venv && source .venv/bin/activate
  2) 安装依赖： 逐条 pip install（参考 [pyproject.toml](pyproject.toml) 中 [project.dependencies]）

启动服务（未安装为包的方式）
- 开发热重载： python -m uvicorn src.model_inference_client.main:app --host 0.0.0.0 --port 6004 --reload
- 普通运行： python -m uvicorn src.model_inference_client.main:app --host 0.0.0.0 --port 6004

启动服务（安装为包后）
- 构建/安装需补充 build-system，当前建议直接使用上面的“未安装方式”。
- 如已安装，可使用入口： [python.run()](src/model_inference_client/main.py:42)

四、配置说明

全局映射
- 模型到后端： [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11) 用于自动推断后端（如 "MAM" -> "triton"）。

Triton 运行器配置 [python.TRITON_RUNNER_CONFIG](src/model_inference_client/config.py:17)
- use_dedicated_server: True 时每个模型独立容器，端口动态分配；False 时使用共享容器，端口固定
- image_name/container_name: Docker 镜像与容器名
- model_repository_host_path: 模型仓库（需要包含每个模型的 config.pbtxt.template）
- runtime_host_path: 运行时挂载目录
- http_port/grpc_port/metrics_port: 端口（共享模式使用）
- gpus: 设备暴露策略（例如 "all" 或 "device=0,1"）
- api_timeout_seconds: 等待 Triton 就绪超时时间
- model_gpu_configs: 每个模型的 instances_per_gpu

ComfyUI 运行器配置
- 需在 [python.COMFYUI_RUNNER_CONFIG](src/model_inference_client/config.py:44) 配置 comfyui_path，指向包含 main.py 的 ComfyUI 目录

五、核心流程与并发控制

启动模型：API -> [python.ModelManager.start_model()](src/model_inference_client/services/model_manager.py:89) -> Runner.start()
- 未指定 model_type 时，自动根据 [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11) 推断
- GPU 独占：在 [python.ModelManager.__init__()](src/model_inference_client/services/model_manager.py:37) 中为每张 GPU 创建 [python.asyncio.Lock()](src/model_inference_client/services/model_manager.py:50)，并用 _active_gpu_models 标记占用
- 在 start 成功前，GPU 锁保持占用；失败会释放并清理状态

停止模型：API -> [python.ModelManager.stop_model()](src/model_inference_client/services/model_manager.py:139) -> Runner.stop()
- 成功停止后释放 GPU 锁

状态查询：API -> [python.ModelManager.get_all_statuses_simplified()](src/model_inference_client/services/model_manager.py:228)

Runner 串行化
- Triton 与 ComfyUI 运行器内部均使用 [python.asyncio.Lock()](src/model_inference_client/runners/triton_runner.py:32) 保证 start/stop 原子性

六、TritonRunner 细节

模式
- 共享模式（use_dedicated_server=False）：首次启动创建共享容器，通过 HTTP API 动态加载/卸载模型
- 独立模式（use_dedicated_server=True）：每个模型独立容器，端口由 [python.find_free_port()](src/model_inference_client/utils/network_utils.py:6) 动态分配

核心方法
- 启动/端口/容器：[python.TritonRunner._start_server()](src/model_inference_client/runners/triton_runner.py:143)
- 生成与刷新配置：[python.generate_triton_config()](src/model_inference_client/utils/config_utils.py:9) · [python.TritonRunner._update_model_config_and_reload()](src/model_inference_client/runners/triton_runner.py:338)
- 加载/卸载模型（HTTP）：[python.TritonRunner._load_model_http()](src/model_inference_client/runners/triton_runner.py:87) · [python.TritonRunner._unload_model_http()](src/model_inference_client/runners/triton_runner.py:101)
- 就绪探测：[python.TritonRunner._wait_for_server_ready()](src/model_inference_client/runners/triton_runner.py:314)
- 状态维护：[python.TritonRunner._update_model_status()](src/model_inference_client/runners/triton_runner.py:372) · [python.TritonRunner.get_all_active_models()](src/model_inference_client/runners/triton_runner.py:530)

模型仓库要求
- 每个模型目录需包含 config.pbtxt.template（示例参见 generate_triton_config 逻辑）
- GPU 分配与实例数由生成的 config.pbtxt 中 instance_group 决定

七、ComfyUIRunner 细节

- 通过子进程启动 ComfyUI 的 main.py，监听本地端口
- 通过 [python.httpx](src/model_inference_client/runners/comfyui_runner.py:13) 轮询 /history 等接口判定就绪
- 重要方法： [python.ComfyUIRunner.start()](src/model_inference_client/runners/comfyui_runner.py:47) · [python.ComfyUIRunner.stop()](src/model_inference_client/runners/comfyui_runner.py:101) · [python.ComfyUIRunner.get_status()](src/model_inference_client/runners/comfyui_runner.py:131)
- 环境变量 CUDA_VISIBLE_DEVICES 控制 GPU 选择

八、API 速览与示例

基础
- 健康检查：GET / 返回 {"status":"ok"}
- 文档：/docs

GPU 与进程
- [python.get_gpus_status()](src/model_inference_client/api/routes.py:22): GET /api/v1/gpus
- [python.kill_process()](src/model_inference_client/api/routes.py:42): DELETE /api/v1/processes/{pid}

模型管理
- [python.start_model_on_gpu()](src/model_inference_client/api/routes.py:64): POST /api/v1/models/start
- [python.stop_model_on_gpu()](src/model_inference_client/api/routes.py:88): POST /api/v1/models/stop
- [python.get_all_models_status()](src/model_inference_client/api/routes.py:117): GET /api/v1/models/status
- [python.get_model_status()](src/model_inference_client/api/routes.py:130): GET /api/v1/models/status/{model_name}
- [python.get_supported_models()](src/model_inference_client/api/routes.py:149): GET /api/v1/models/supported

示例请求（curl）
- 启动 Triton 模型：
  curl -X POST http://localhost:6004/api/v1/models/start -H "Content-Type: application/json" -d '{"model_name":"MAM","gpu_id":0,"config":{}}'
- 停止模型：
  curl -X POST http://localhost:6004/api/v1/models/stop -H "Content-Type: application/json" -d '{"model_name":"MAM","gpu_id":0}'
- 查看状态：
  curl http://localhost:6004/api/v1/models/status

九、部署与运维

- 前置：Docker 可用、NVIDIA 驱动、模型仓库与 runtime 目录挂载路径与权限正确
- 端口：共享模式使用配置端口；独立模式端口动态分配（注意冲突与防火墙）
- 日志：容器日志（docker logs）、应用标准输出、Triton HTTP 返回值
- 关停：优先调用 API 停止模型；应用退出时由 [python.model_manager.shutdown()](src/model_inference_client/services/model_manager.py:294) 协调 Runner 关停

十、排障与常见问题

- NVML 未安装或无 GPU：/gpus 端点会 501 或返回 0 张卡；按需安装 nvidia-ml-py 与驱动
- Docker 不可用：Triton 无法启动容器；请检查权限与 daemon
- 模型缺少 config.pbtxt.template：Triton 加载失败；请补齐模板
- 端口被占用：共享模式固定端口，如被占用需调整配置；独立模式使用 [python.find_free_port()](src/model_inference_client/utils/network_utils.py:6)，仍可能与安全策略冲突
- GPU 独占冲突：同一 GPU 同时仅允许一个模型，见 [python.ModelManager.start_model()](src/model_inference_client/services/model_manager.py:106) 的占用检查
- BaseRunner 同步签名提示：当前 Runner 为异步实现，扩展 Runner 时请遵循异步接口（见 Triton/ComfyUI 实现）

十一、扩展新的后端 Runner

步骤
1) 在 runners/ 下新增文件并实现异步接口： start/stop/get_status（参考 [python.ComfyUIRunner.start()](src/model_inference_client/runners/comfyui_runner.py:47)、[python.TritonRunner.start()](src/model_inference_client/runners/triton_runner.py:409)）
2) 在 [python.ModelManager._init_runners()](src/model_inference_client/services/model_manager.py:55) 注册
3) 在 [python.MODEL_BACKEND_MAPPING](src/model_inference_client/config.py:11) 中维护模型与后端映射
4) 若需要 API 兼容，只需复用现有请求体 [python.StartRequest()](src/model_inference_client/models/schemas.py:30)/[python.StopRequest()](src/model_inference_client/models/schemas.py:40)

重要注意
- 使用 asyncio.Lock 保证 Runner 内部串行
- 遵循 ModelInstanceInfo 字段约定，保持 /status 语义稳定
- 长耗时操作（容器启动、模型加载）必须非阻塞（异步请求+轮询）

十二、质量保障与发布

- 最小化集成测试建议
  - Mock NVML：覆盖 /gpus 分支
  - Triton API：针对 _http_request/_load/_unload 做集成或契约测试
  - ModelManager 并发：多协程同时 start/stop 同一 GPU，验证互斥
- Lint/格式：Black/ruff（可选）
- 发布：当前缺少 [build-system] 配置，推荐以源码/容器方式交付；如需发布 PyPI，请补充构建配置（hatchling 或 setuptools）

十三、变更记录约定（建议）

- 每次新增 Runner 或模型，请更新： [README.md](README.md) 与本文档“配置/扩展”章节
- 对并发/锁逻辑改动，请在 PR 中描述并附并发测试结果

附录：关键类型与字段速查

- [python.StartRequest()](src/model_inference_client/models/schemas.py:30): model_name, model_type?, gpu_id, config
- [python.StopRequest()](src/model_inference_client/models/schemas.py:40): model_name, model_type?, gpu_id, instance_id?
- [python.ModelInstanceInfo()](src/model_inference_client/models/schemas.py:50): instance_id, model_name, model_type, active_gpu_ids, total_instances, status, pid, port, created_at, updated_at
- [python.ModelStatusItem()](src/model_inference_client/models/schemas.py:69): model_name, model_type, gpu_id

结束语

若有部署环境、CI/CD、监控系统（如 Prometheus/Grafana）接入，请在本文件基础上补充对应章节，以确保交接完整与可运维性。