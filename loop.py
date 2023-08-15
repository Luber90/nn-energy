from subprocess import call




for i in range(9):
    if call(f"./run_network.sh 60000 True 10 False 32", shell=True) != 0:
        exit()