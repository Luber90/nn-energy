from subprocess import call

numbers = (30_000, 45_000, 60_000)
bools = [True]

for n in numbers:
    for b in bools:
        for i in range(5):
            if call(f"./run_network.sh {n} {b} 5", shell=True) != 0:
                exit()
