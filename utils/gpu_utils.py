import os
import signal
from typing import List, Dict, Any

try:
    import pynvml
except ImportError:
    pynvml = None

def get_gpu_info() -> List[Dict[str, Any]]:
    if pynvml is None:
        raise ImportError("pynvml is not installed. Please install it with 'pip install nvidia-ml-py'")

    pynvml.nvmlInit()
    gpus = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
        
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError_NotSupported:
            procs = []
        
        processes_info = []
        for p in procs:
            try:
                # This can fail if the process terminates between calls
                proc_name = pynvml.nvmlSystemGetProcessName(p.pid)
                if isinstance(proc_name, bytes):
                    name = proc_name.decode('utf-8')
                else:
                    name = proc_name
            except (pynvml.NVMLError_NotFound, pynvml.NVMLError_Unknown, pynvml.NVMLError_GpuIsLost):
                name = "N/A"

            processes_info.append({
                "pid": p.pid,
                "name": name,
                "gpu_memory_usage": p.usedGpuMemory // (1024**2) # Convert to MB
            })
            
        gpu_info = {
            "id": i,
            "memory_usage": mem_info.used / mem_info.total * 100,
            "power_draw": power_usage,
            "processes": processes_info
        }
        gpus.append(gpu_info)

    pynvml.nvmlShutdown()
    return gpus

def kill_process_by_pid(pid: int) -> Dict[str, Any]:
    try:
        os.kill(pid, signal.SIGKILL)
        return {"pid": pid, "status": "killed"}
    except ProcessLookupError:
        return {"pid": pid, "status": "error", "message": "Process not found"}
    except PermissionError:
        return {"pid": pid, "status": "error", "message": "Permission denied"}
    except Exception as e:
        return {"pid": pid, "status": "error", "message": str(e)} 