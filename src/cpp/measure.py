import os
from time import sleep
import socket
from prometheus_client import Gauge, start_http_server
import prometheus_client
from pynvml.smi import nvidia_smi
import pynvml as nvml

memory_gauge = Gauge('network_gpu_memory_usage_mb', 'GPU memory used by neural network in MiB')
usage_gauge = Gauge('gpu_usage', 'GPU usage percentage')

def run(PID):
    nvml.nvmlInit()
    gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)

    nvsmi = nvidia_smi.getInstance()

    start_http_server(8000)

    SLEEP_TIME = 10.0

    print(f'Measuring process {PID}')

    while True:
        gpu_memory_usage = 0.0
        for process in nvml.nvmlDeviceGetComputeRunningProcesses_v3(gpu_handle):
            if process.pid == PID:
                gpu_memory_usage = process.usedGpuMemory/(1024*1024)
        memory_gauge.set(gpu_memory_usage)
        
        gpu_usage = nvsmi.DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
        usage_gauge.set(gpu_usage)


        sleep(SLEEP_TIME)

if __name__ == '__main__':
    run()


