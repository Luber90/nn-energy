from subprocess import call

numbers = [15_000, 30_000, 60_000, 75_000]

for n in numbers:
    for i in range(10):
        if call(f"./run_network.sh {n} False 5 False 32", shell=True) != 0:
            exit()
