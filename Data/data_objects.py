import numpy as np
from torchvision import datasets
from Data.utils import create_task_composition, load_task_with_labels
import torchvision.transforms as transforms

class CIFAR_10_C:
    def __init__(self, args, root='/root/Experiments/MobiCom24_Delta/Data/RawData'):
        """
        Initialize the CIFAR-10-C dataset.
        
        Args:
            args: Configuration arguments.
            root: Root directory for the dataset.
        """
        self.root = root
        self.dataset = 'cifar-10-C'
        self.args = args
    
    def setup(self, context_list=[], num_tasks=5, fixed_order=False):
        """
        Set up the dataset based on context list.
        
        Args:
            context_list: List of contexts.
            num_tasks: Number of tasks.
            fixed_order: Whether to use a fixed order for tasks.
        """
        context_list = context_list.split('+')
        x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
            
        for context in context_list:
            style, level = context[:-1], int(context[-1])
            x = np.load(f'{self.root}/cifar-10-C/{style}.npy')
            y = np.load(f'{self.root}/cifar-10-C/labels.npy')
            offset = 10000 * (level - 1)
            # Divide the fist 2K samples into training set and testing set
            # Device-side data will be sampled from training set
            x_test, y_test = x[offset:offset + 1000], y[offset:offset + 1000]
            x_train, y_train = x[offset + 1000:offset + 2000], y[offset + 1000:offset + 2000]
                
            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_test_list.append(x_test)
            y_test_list.append(y_test)
            
        self.task_labels = create_task_composition(class_nums=10, num_tasks=num_tasks, fixed_order=fixed_order, context_list=context_list)
        self.train_set, self.test_set = [], []
            
        for labels in self.task_labels:
            if len(context_list) == 1:
                context_id = 0
            else:
                context_id = self.task_labels.index(labels)
                
            x_test_tmp, y_test_tmp = load_task_with_labels(x_test_list[context_id], y_test_list[context_id], labels)
            x_train_tmp, y_train_tmp = load_task_with_labels(x_train_list[context_id], y_train_list[context_id], labels)
                
            self.train_set.append((x_train_tmp, y_train_tmp))
            self.test_set.append((x_test_tmp, y_test_tmp))
            
        del x_train_list, y_train_list, x_test_list, y_test_list

    def new_task(self, cur_task):
        """
        Get the training data and labels for the current task.
        
        Args:
            cur_task: Current task index.
        
        Returns:
            x_train: Training data.
            y_train: Training labels.
            labels: Unique labels in the training data.
        """
        x_train, y_train = self.train_set[cur_task]
        labels = set(y_train)
        return x_train, y_train, labels
    
    def new_run(self):
        """
        Set up a new run based on the configuration arguments.
        """
        self.setup(self.args.context_list, self.args.num_tasks, self.args.fixed_order)
        
    def get_test_set(self):
        """
        Get the test set.
        
        Returns:
            test_set: Test data and labels.
        """
        return self.test_set
    

class HAR:
    def __init__(self, args, root='/root/Experiments/MobiCom24_Delta/Data/RawData/har'):
        """
        Initialize the HAR dataset.
        
        Args:
            args: Configuration arguments.
            root: Root directory for the dataset.
        """
        self.root = root
        self.dataset = 'har'
        self.args = args
        self.download_load()
        
    def download_load(self):
        """
        Download and load the HAR dataset.
        """
        datasets = ["hhar", "uci", "motion", "shoaib"]
        self.train_data, self.train_label, self.train_user = {}, {}, {}
        
        for dataset in datasets:
            # device-side data will be further divided into training set and testing set
            dataset_train = np.load(f'{self.root}/{dataset}_device.npz')
            self.train_data[dataset] = dataset_train['arr_0']
            self.train_label[dataset] = dataset_train['arr_1']
            self.train_user[dataset] = dataset_train['arr_2']
            print(dataset, np.unique(self.train_label[dataset]))
    
    def setup(self, context_list=[], num_tasks=5, fixed_order=False):
        """
        Set up the dataset based on context list.
        
        Args:
            context_list: List of contexts.
            num_tasks: Number of tasks.
            fixed_order: Whether to use a fixed order for tasks.
        """
        context_list = context_list.split('+')
        x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
            
        for context in context_list:
            dataset, user_id = context.split('_')
            x_tmp, y_tmp, user_tmp = self.train_data[dataset], self.train_label[dataset], self.train_user[dataset]
            idx = np.where(user_tmp == int(user_id))[0]
            np.random.shuffle(idx)
                
            split_point = len(idx) // 3 
            x_train_list.append(x_tmp[idx[:split_point]])
            y_train_list.append(y_tmp[idx[:split_point]])
            x_test_list.append(x_tmp[idx[split_point:]])
            y_test_list.append(y_tmp[idx[split_point:]])
            
        self.task_labels = create_task_composition(class_nums=6, num_tasks=num_tasks, fixed_order=fixed_order, context_list=context_list)
        self.train_set, self.test_set = [], []
            
        for labels in self.task_labels:
            if len(context_list) == 1:
                context_id = 0
            else:
                context_id = self.task_labels.index(labels)
                
            x_test_tmp, y_test_tmp = load_task_with_labels(x_test_list[context_id], y_test_list[context_id], labels)
            x_train_tmp, y_train_tmp = load_task_with_labels(x_train_list[context_id], y_train_list[context_id], labels)
                
            self.train_set.append((x_train_tmp, y_train_tmp))
            self.test_set.append((x_test_tmp, y_test_tmp))
            
        del x_train_list, y_train_list, x_test_list, y_test_list
    
    def new_task(self, cur_task):
        x_train, y_train = self.train_set[cur_task]
        labels = set(y_train)
        return x_train, y_train, labels
    
    def new_run(self, ):
        self.setup(self.args.context_list, self.args.num_tasks, self.args.fixed_order)
        
    def get_test_set(self):
        return self.test_set      
    
class TextClassification:
    def __init__(self, args, root='/root/Experiments/MobiCom24_Delta/Data/RawData/textclassification'):
        """
        Initialize the TextClassification dataset.
        
        Args:
            args: Configuration arguments.
            root: Root directory for the dataset.
        """
        self.root = root
        self.dataset = 'textclassification'
        self.args = args
        self.download_load()
        
    def download_load(self):
        """
        Download and load the TextClassification dataset.
        """
        self.datasets_device, self.datasets_cloud = {}, {}
        for dataset in ["de", "en", "es", "fr", 'ru']:
            self.datasets_device[dataset] = np.load(f'{self.root}/{dataset}_device.npy', allow_pickle=True).item()
            self.datasets_cloud[dataset] = np.load(f'{self.root}/{dataset}_cloud.npy', allow_pickle=True).item()

    def setup(self, context_list=[], num_tasks=5, fixed_order=False):
        """
        Set up the dataset based on context list.
        
        Args:
            context_list: List of contexts.
            num_tasks: Number of tasks.
            fixed_order: Whether to use a fixed order for tasks.
        """
        context_list = context_list.split('+')
        x_train_list, y_train_list, mask_train_list, x_test_list, mask_test_list, y_test_list = [], [], [], [], [], []
            
        for context in context_list:
            dataset = context
            x_tmp, mask_tmp, y_tmp = self.datasets_device[dataset]['inputs'], self.datasets_device[dataset]['masks'], self.datasets_device[dataset]['labels']
            idx = np.arange(len(x_tmp))
            np.random.shuffle(idx)
                
            split_point = int(0.6 * len(idx))
            x_train_list.append(x_tmp[idx[:split_point]])
            mask_train_list.append(mask_tmp[idx[:split_point]])
            y_train_list.append(y_tmp[idx[:split_point]])
            x_test_list.append(x_tmp[idx[split_point:]])
            mask_test_list.append(mask_tmp[idx[split_point:]])
            y_test_list.append(y_tmp[idx[split_point:]])
            
        self.task_labels = create_task_composition(class_nums=10, num_tasks=num_tasks, fixed_order=fixed_order, context_list=context_list)
        self.train_set, self.test_set = [], []
            
        for labels in self.task_labels:
            if len(context_list) == 1:
                context_id = 0
            else:
                context_id = self.task_labels.index(labels)
                
            x_test_tmp, mask_test_tmp, y_test_tmp = load_task_with_labels(x_test_list[context_id], y_test_list[context_id], labels, masks=mask_test_list[context_id])
            x_train_tmp, mask_train_tmp, y_train_tmp = load_task_with_labels(x_train_list[context_id], y_train_list[context_id], labels, masks=mask_train_list[context_id])
                
            self.train_set.append((x_train_tmp, mask_train_tmp, y_train_tmp))
            self.test_set.append((x_test_tmp, mask_test_tmp, y_test_tmp))
            
        del x_train_list, mask_train_list, y_train_list, x_test_list, mask_test_list, y_test_list
          
    def new_task(self, cur_task):
        x_train, masks_train, y_train = self.train_set[cur_task]
        labels = np.unique(y_train)
        return x_train, masks_train, y_train, labels
    
    def new_run(self, ):
        self.setup(self.args.fontext_list, self.args.num_tasks, self.args.fixed_order)
        
    def get_test_set(self):
        return self.test_set      

    def load_task_with_labels(x, mask, y, labels, num_per_label=10000):
        tmp = []
        for i in labels:
            tmp.append((np.where(y==i)[0][:num_per_label]))
        idx = np.concatenate(tmp, axis=None)
        return x[idx], mask[idx], y[idx]
    
data_objects = {
    'cifar-10-C': CIFAR_10_C,
    'har': HAR,
    'textclassification': TextClassification
}