import matplotlib.pyplot as plt
import numpy as np

def draw_subplot(subid, title, xlabel, ylabel, results, y_min=None, alt_labels=None):   
    plt.subplot(subid)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



    m1 = np.mean([v for v in results.values()], axis=1)

    for i, mean in enumerate(m1):
        text = ' μ={:.4f}'.format(m1[i])
        print(text)
        #plt.text(i+0.5, m1[i],text)

    if y_min is not None:
        y_max = np.round(np.max(np.ravel([v for v in results.values()]))+200, decimals=-2)
        axis = plt.axis([0.5, len(results.keys())+0.5, y_min, y_max])
    violin = plt.violinplot([v for v in results.values()], showmeans=True)
    for pc in violin['bodies']:
        pc.set_alpha(0.8)
    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin[partname]
        vp.set_edgecolor('blue')
    for i, partname in enumerate(['cmeans']):
        vp = violin[partname]
        vp.set_edgecolor('red')
    x_pos = np.arange(1, len(results.keys())+1)
    print(results.keys())
    if alt_labels is None:
        plt.xticks(x_pos, results.keys())
    else:
        print(x_pos)
        print(alt_labels)
        plt.xticks(x_pos, alt_labels)

def draw_graphs_c(stat_id, stat_name, file_name='results.txt', mini=4000, alt_labels=None):
    results_cpu = {}
    results_gpu = {}
    results_memory = {}
    results_cpu_memory = {}
    with open(file_name, "r") as file:
        for exp in file:
            stats = exp.split(';')
            if isinstance(stat_id, tuple):
                category = stats[stat_id[0]]+" "+stats[stat_id[1]]
            else:    
                category = stats[stat_id]
            #print(category)

            watt_start = exp.find(',') + 2
            if results_cpu.get(category) is None:
                results_cpu[category] = [float(stats[-4])]
                results_gpu[category] = [float(stats[-3])]
                results_memory[category] = [float(stats[-2])]
                results_cpu_memory[category] = [float(stats[-1])]
            else:
                results_cpu[category].append(float(stats[-4]))
                results_gpu[category].append(float(stats[-3]))
                results_memory[category].append(float(stats[-2]))
                results_cpu_memory[category].append(float(stats[-1]))

    fig = plt.figure(figsize=(13, 10))


    grid = plt.GridSpec(2, 6, wspace=0.6, hspace=0.5, bottom=0.05, top=0.95, left=0.07, right=0.93)

    draw_subplot(grid[0, 0:3], 'CPU Wh', stat_name, 'Wh', results_cpu, alt_labels=alt_labels)
    draw_subplot(grid[0, 3:], 'GPU Wh', stat_name, 'Wh', results_gpu, alt_labels=alt_labels)
    draw_subplot(grid[1, 0:3], 'GPU Memory used', stat_name, 'MiB', results_memory, y_min=mini, alt_labels=alt_labels)
    draw_subplot(grid[1, 3:], 'CPU Memory used', stat_name, 'MiB', results_cpu_memory, y_min=3000, alt_labels=alt_labels)

    plt.savefig('figure.png')
    plt.show()

def draw_graphs(stat_id, stat_name, file_name='results.txt', mini=4000, alt_labels=None):
    results_cpu = {}
    results_gpu = {}
    results_memory = {}
    results_cpu_memory = {}
    results_loss = {}
    results_ssim = {}
    results_psnr = {}
    with open(file_name, "r") as file:
        for exp in file:
            stats = exp.split(';')
            if isinstance(stat_id, tuple):
                category = stats[stat_id[0]]+" "+stats[stat_id[1]]
            else:    
                category = stats[stat_id]
            #print(category)

            watt_start = exp.find(',') + 2
            if results_cpu.get(category) is None:
                results_cpu[category] = [float(stats[-4])]
                results_gpu[category] = [float(stats[-3])]
                results_memory[category] = [float(stats[-2])]
                results_cpu_memory[category] = [float(stats[-1])]
                results_loss[category] = [float(stats[-7])]
                results_ssim[category] = [float(stats[-6])]
                results_psnr[category] = [float(stats[-5])]
            else:
                results_cpu[category].append(float(stats[-4]))
                results_gpu[category].append(float(stats[-3]))
                results_memory[category].append(float(stats[-2]))
                results_cpu_memory[category].append(float(stats[-1]))
                results_loss[category].append(float(stats[-7]))
                results_ssim[category].append(float(stats[-6]))
                results_psnr[category].append(float(stats[-5]))
    #print(results_cpu)
    #print(results_gpu)

    fig = plt.figure(figsize=(13, 10))


    grid = plt.GridSpec(3, 6, wspace=0.6, hspace=0.5, bottom=0.05, top=0.95, left=0.07, right=0.93)

    draw_subplot(grid[0, 1:3], 'CPU Wh', stat_name, 'Wh', results_cpu, alt_labels=alt_labels)
    draw_subplot(grid[0, 3:5], 'GPU Wh', stat_name, 'Wh', results_gpu, alt_labels=alt_labels)
    draw_subplot(grid[1, 1:3], 'GPU Memory used', stat_name, 'MiB', results_memory, y_min=mini, alt_labels=alt_labels)
    draw_subplot(grid[1, 3:5], 'CPU Memory used', stat_name, 'MiB', results_cpu_memory, y_min=3000, alt_labels=alt_labels)
    draw_subplot(grid[2, 0:2], 'Validation loss', stat_name, 'Loss', results_loss, alt_labels=alt_labels)
    draw_subplot(grid[2, 2:4], 'Validation SSim', stat_name, 'SSIM', results_ssim, alt_labels=alt_labels)
    draw_subplot(grid[2, 4:], 'Validation PSNR', stat_name, 'PSNR', results_psnr, alt_labels=alt_labels)

    plt.savefig('figure.png')
    plt.show()


if __name__ == "__main__":
    #draw_graphs(4, 'Number of epochs', 'results_epochs.txt', mini=6000)
    #draw_graphs(2, 'Size of the set', 'results_size.txt', mini=6000)
    #draw_graphs((4, 3), 'Size of the network', 'results_net_size.txt', mini=6000, alt_labels=('Bigger', 'Smaller'))
    #draw_graphs(2, 'Size of the set', 'results_size_old.txt')
    #draw_graphs(3, 'Size of the network', 'results_net_size_old.txt')
    #draw_graphs(5, 'Automatic mixed precision', 'results_mixed.txt', alt_labels=['Full precision', 'Mixed precision'])
    #draw_graphs(6, 'Batch size', 'results_batch.txt', mini=6000)
    draw_graphs_c(2, 'Programming language', 'results_c.txt', mini=6000, alt_labels=('Python', 'C++'))
