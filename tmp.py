from pynvml.smi import nvidia_smi
import pynvml as nvml

nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)
a = nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
print(a[0].usedGpuMemory/(1024*1024))