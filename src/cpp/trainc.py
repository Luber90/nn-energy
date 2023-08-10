import measure
import threading
import signal
import sys
from prometheus_api_client import PrometheusConnect, MetricsList, Metric
from prometheus_api_client.utils import parse_datetime
import datetime
import os
import subprocess


if __name__ == '__main__':
    prom = PrometheusConnect(url="http://172.17.0.1:9090", disable_ssl=True)

    start_time = str(datetime.datetime.now())[:-7]
    proc = subprocess.Popen(["./src/cpp/build/networkc"])
    measure_thread = threading.Thread(target=lambda: measure.run(proc.pid))
    measure_thread.daemon = True
    measure_thread.start()
    proc.wait()
    end_time = str(datetime.datetime.now())[:-7]

    start_time = parse_datetime(start_time)
    end_time = parse_datetime(end_time)

    metric_data = prom.custom_query_range(
        query='sum(scaph_process_power_consumption_microwatts{cmdline=~".*networkc.*"}/1000000)*0.00278',
        start_time=start_time,
        end_time=end_time,
        step=10.0
    )

    cpu_W = sum(float(v[1]) for v in metric_data[0]['values'])


    metric_data = prom.custom_query_range(
        query='gpu_usage*0.00278',
        start_time=start_time,
        end_time=end_time,
        step=10.0
    )

    gpu_W = sum(float(v[1]) for v in metric_data[0]['values'])

    metric_data = prom.custom_query_range(
        query='network_gpu_memory_usage_mb',
        start_time=start_time,
        end_time=end_time,
        step=10.0
    )

    metric_data = [float(v[1]) for v in metric_data[0]['values']]
    gpu_mem = sum(metric_data)/len(metric_data)

    # with open("results.txt", "a") as file:
    #     file.write(f"{start_time};{end_time};{number_of_images};{smaller};{epochs};{autocast};{batch_size};{adam};{val_loss:.5f};{val_ssim:.5f};{val_psnr:.5f};{cpu_W:.2f};{gpu_W:.2f};{gpu_mem:.2f}\n")
    
    print(cpu_W, gpu_W, gpu_mem)

