import matplotlib.pyplot as plt


if __name__ == "__main__":
    results_cpu = {}
    results_gpu = {}
    with open("results.txt", "r") as file:
        for exp in file:
            stats = exp.split(';')
            num_images = stats[2]
            print(num_images)

            watt_start = exp.find(',') + 2
            if results_cpu.get(num_images) is None:
                results_cpu[num_images] = [float(stats[-3])]
                results_gpu[num_images] = [float(stats[-2])]
            else:
                results_cpu[num_images].append(float(stats[-3]))
                results_gpu[num_images].append(float(stats[-2]))
    print(results_cpu)
    print(results_gpu)

    plt.subplot(211)
    plt.boxplot([v for v in results_cpu.values()], labels=[k for k in results_cpu.keys()])

    plt.subplot(212)
    plt.boxplot([v for v in results_gpu.values()], labels=[k for k in results_gpu.keys()])
    plt.show()