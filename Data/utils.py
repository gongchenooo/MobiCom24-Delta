import numpy as np
import torch
from torch.utils import data

def create_task_composition(class_nums, num_tasks, fixed_order=False, context_list=[]):
    classes_per_task = class_nums // num_tasks
    total_classes = classes_per_task * num_tasks
    # assert total_classes == class_nums
    label_array = np.arange(0, total_classes)
    if not fixed_order:
        np.random.shuffle(label_array)
    task_labels = []
    for tt in range(num_tasks):
        tt_offset = tt * classes_per_task
        task_labels.append(list(label_array[tt_offset:tt_offset+classes_per_task]))
        context = context_list[0] if len(context_list)==1 else context_list[tt]
        print('Task {},\tLabels:{}\tContext:{}'.format(tt, task_labels[tt], context))
    return task_labels

def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y==i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]
def load_task_with_labels(x, y, labels, num_per_label=10000, masks=None):
    tmp = []
    for i in labels:
        tmp.append((np.where(y==i)[0][:num_per_label]))
    idx = np.concatenate(tmp, axis=None)
    if masks is None:
        return x[idx], y[idx]
    else:
        return x[idx], masks[idx], y[idx]

