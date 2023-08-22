from subprocess import call

for i in range(9):
    if call(f"./run_networkc.sh", shell=True) != 0:
        exit()