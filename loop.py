from subprocess import call

for i in range(9):
    if call(f"./run_network.sh 45000 False 5 False 32", shell=True) != 0:
        exit()

for i in range(10):
    if call(f"./run_network.sh 30000 False 5 False 32", shell=True) != 0:
        exit()