import matplotlib.pyplot as plt
import numpy as np

def read_file(path, num_task, num_run):
    f = open(path, 'r')
    acc_array = [[] for _ in range(num_task)]
    task_iter = []
    current_run = 0

    for line in f:
        parts = line.split()
        if 'Run:' in parts:
            current_run += 1
            if current_run > num_run:
                current_run -= 1
                break
            continue
        if 'Test' in parts:
            for i in range(2, 2+num_task):
                start = 1 if i==2 else 0
                end = -1
                acc_array[i-2].append(float(parts[i][start:end]))
        elif 'Task:' in parts:
            task_iter.append(len(acc_array[0]))
    num_run = current_run
        
    
    # Divide for each run
    acc_array_new = [[] for _ in range(num_run)]
    for r in range(num_run):
        for t in range(num_task):
            acc_tmp = acc_array[t][r * len(acc_array[t]) // num_run:(r + 1)*len(acc_array[t]) // num_run]
            acc_array_new[r].append(acc_tmp)
    acc_array = np.array(acc_array_new) # [num_run, num_task, num_iter]
    return task_iter[:num_task], acc_array

def smooth(arr, window_size=1):
    if window_size == 0:
        return arr
    new_arr = []
    if isinstance(arr, np.ndarray) and len(arr.shape) == 2:
        for row in arr:
            new_row = []
            for j, element in enumerate(row):
                start_idx = max(0, j - window_size)
                end_idx = min(arr.shape[1], j + window_size + 1)
                new_row.append(np.mean(row[start_idx:end_idx]))
            new_arr.append(new_row)
    else:
        for i, element in enumerate(arr):
            start_idx = max(0, i - window_size)
            end_idx = min(len(arr), i + window_size + 1)
            new_arr.append(np.mean(arr[start_idx:end_idx]))
    return np.array(new_arr)

def find_more(lst, value):
    for index, element in enumerate(lst):
        if element >= value:
            return index
    return len(lst)

dataset_info = {
    "cifar-10-C": {
        "context_list": ["fog5", ],
        # "context_list": ["fog5+gaussian_noise5+glass_blur5+spatter5+pixelate5", ],
        "method_list": ["original", "random", "delta"],
        "model": "ResNet18_pretrained",
        "optimizer": "SGD",
        "num_per_task": 10,
        "enrich_num": 100,
        "lr": 0.01,
        "num_task": 5,
        "num_run": 2,
        "cluster_num": 1,
        "cluster_topK": 1,
        "temp": 1.0,
        "seed": 1
    }
}

def plot_overall(dataset):
    info = dataset_info[dataset]
    
    for context in info["context_list"]:
        plt.clf()
        root = f"/root/Experiments/MobiCom24_Delta/Log/{dataset}/" + \
               f"{info['model']}_{info['optimizer']}_{info['lr']}/" + \
               f"#Tasks={info['num_task']}_Context={context}/"
        for method in info["method_list"]:
            if method in ["original", "random"]:
                path = root + f"test_{info['num_per_task']}/{method}_K-Means1_{info['enrich_num']}_{info['seed']}.txt"
            elif method == "delta":
                path = root + f"test_{info['num_per_task']}/{method}_K-Means{info['cluster_num']}_{info['cluster_topK']}_" + \
                       f"{info['temp']}_{info['enrich_num']}_{info['seed']}.txt"
            elif method in ["fskd", "fsro", "fspr"]:
                path = root + f"{method}_{info['num_per_task']}/original_K-Means1_{info['enrich_num']}_{info['seed']}.txt"
            task_iter, acc_array = read_file(path=path, num_run=info['num_run'], num_task=info['num_task'])
            acc_mean = smooth(np.mean(acc_array, axis=0), 0) # [num_task, num_iter]
            print("Method: {}\tAvg_Acc: {:.2f} [{}]".format(
                method, acc_mean.mean(0)[-10:].max()*100, (acc_array.mean(1)[:, -10:]).max(1)))
            plt.plot(acc_mean.mean(0)*100, label=method, lw=2, marker="o", markevery=4)
        x = np.arange(0, acc_mean.shape[1], 25)
        plt.xticks(x, x*20, fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel("Local Training Iterations", fontsize=14)
        plt.ylabel("Avg. Acc. on All Contexts", fontsize=14)
        plt.legend(fontsize=13)
        plt.grid(linestyle=":")
        plt.tight_layout()
        plt.savefig(f"Figures/Overall/{dataset}_{context}.png")

plot_overall("cifar-10-C")