import matplotlib.pyplot as plt
import numpy as np

def draw_subplot(subid, title, xlabel, ylabel, results, y_min=None):   
    plt.subplot(subid)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_min is not None:
        y_max = np.round(np.max(np.ravel([v for v in results.values()]))+200, decimals=-2)
        axis = plt.axis([0.5, len(results.keys())+0.5, y_min, y_max])
    violin = plt.violinplot([v for v in results.values()])
    for pc in violin['bodies']:
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin[partname]
        vp.set_edgecolor('blue')
    x_pos = np.arange(1, len(results.keys())+1)
    plt.xticks(x_pos, results.keys())

def draw_graphs(stat_id, stat_name, file_name='results.txt', mini=4000):
    results_cpu = {}
    results_gpu = {}
    results_memory = {}
    results_loss = {}
    results_ssim = {}
    results_psnr = {}
    with open(file_name, "r") as file:
        for exp in file:
            stats = exp.split(';')
            category = stats[stat_id]
            #print(category)

            watt_start = exp.find(',') + 2
            if results_cpu.get(category) is None:
                results_cpu[category] = [float(stats[-3])]
                results_gpu[category] = [float(stats[-2])]
                results_memory[category] = [float(stats[-1])]
                results_loss[category] = [float(stats[-6])]
                results_ssim[category] = [float(stats[-5])]
                results_psnr[category] = [float(stats[-4])]
            else:
                results_cpu[category].append(float(stats[-3]))
                results_gpu[category].append(float(stats[-2]))
                results_memory[category].append(float(stats[-1]))
                results_loss[category].append(float(stats[-6]))
                results_ssim[category].append(float(stats[-5]))
                results_psnr[category].append(float(stats[-4]))
    #print(results_cpu)
    #print(results_gpu)

    plt.figure(figsize=(15, 8), constrained_layout=True)

    draw_subplot(231, 'CPU Wh', stat_name, 'Wh', results_cpu)
    draw_subplot(232, 'GPU Wh', stat_name, 'Wh', results_gpu)
    draw_subplot(233, 'GPU Memory used', stat_name, 'MiB', results_memory, y_min=mini)
    draw_subplot(234, 'Validation loss', stat_name, 'Loss', results_loss)
    draw_subplot(235, 'Validation SSim', stat_name, 'SSim', results_ssim)
    draw_subplot(236, 'Validation PSNR', stat_name, 'PSNR', results_psnr)

    plt.savefig('figure.png')
    plt.show()


if __name__ == "__main__":
    #draw_graphs(4, 'Number of epochs', 'results_epochs.txt', mini=6000)
    #draw_graphs(2, 'Size of the set', 'results_size.txt')
    draw_graphs(3, 'Size of the network', 'results_net_size.txt', mini=6000)
    #draw_graphs(2, 'Size of the set', 'results_size_old.txt')
    #draw_graphs(3, 'Size of the network', 'results_net_size_old.txt')
    #draw_graphs(5, 'Automatic mixed precision', 'results_mixed.txt')
    #draw_graphs(6, 'Batch size', 'results_batch.txt', mini=6000)
