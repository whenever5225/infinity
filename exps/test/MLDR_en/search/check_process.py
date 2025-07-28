import os
import signal
import subprocess
import time
import psutil

import psutil
def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f" {file_path} ")
    except Exception as e:
        print(f" - {e}")
    return None


def start_service_and_kill(command):
    try:
        # 
        process = subprocess.Popen(command, shell=True)
        print(f" ID: {process.pid}")

        # 
        time.sleep(5)

        # 
        process.terminate()
        print(f" {process.pid} ")
    except Exception as e:
        print(f" - {e}")

def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass

if __name__ == "__main__":
    # 
    file_path = '/home/ubuntu/hybridsearch/experiments/query_memory_file'
    content = read_file_content(file_path)


    # 
    service_command = "/home/ubuntu/hybridsearch/cmake-build-release/src/hybridsearch -f /home/ubuntu/hybridsearch/conf/hybridsearch_conf.toml"  #  HTTP 
    process = subprocess.Popen(service_command, shell=True)
    time.sleep(3)
    print(f" ID: {process.pid}")
    print(__file__)
    time.sleep(5)
    terminate_process_tree(process.pid)
    print(f" {process.pid} ")
    