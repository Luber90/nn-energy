import os
from time import sleep
import socket
from prometheus_client import Gauge, start_http_server

# def check_pid(PID):
#     try:
#         os.kill(PID, 0)
#     except OSError:
#         return False
#     return True

memory_gauge = Gauge('network_gpu_memory_usage_mb', 'GPU memory used by neural network in MiB')
usage_gauge = Gauge('gpu_usage', 'GPU usage percentage')

def run():
    start_http_server(8000)

    IP = '127.0.0.1'
    PORT = 5005

    SLEEP_TIME = 1.5
    PID = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((IP, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1024)
            PID = int(data.decode())

    print(f'Measuring process {PID}') 

    while True:
        with os.popen('nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv') as stream:
            result = stream.read().split('\n')
            if len(result) > 2:
                gpu_memory_usage = 0.0
                for i in result[1:]:
                    if i.split(', ')[0] == str(PID):
                        gpu_memory_usage = float(i.split(', ')[1][:-3])
                memory_gauge.set(gpu_memory_usage)
        
        with os.popen('nvidia-smi --query-gpu=power.draw --format=csv') as stream:
            gpu_usage = float(stream.read().split('\n')[1][:-2])
            usage_gauge.set(gpu_usage)


        sleep(SLEEP_TIME)
    print('Measuring ended.')

if __name__ == '__main__':
    run()


