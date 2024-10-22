import torchvision.transforms as transforms
# from dataset.cifar10 import CIFAR10
# from dataset.har import HAR
# from dataset.speechcommand import SpeechCommand
# from dataset.textclassification import TextClassification
import torch
from Models.resnet import Reduced_ResNet18, ResNet18, ResNet34, ResNet50, ResNet101
from Models.pretrain import ResNet18_pretrained, ResNet101_pretrained, DCNN_pretrained, VGG11_pretrained, Bert_pretrained
from torch.utils import data
import pynvml

MEMORY_UNIT = 1024*1024

input_size_match = {
    'cifar10': [3, 224, 224],
    'cifar-10-C': [3, 224, 224],
    'har': [20, 6],
    'speechcommand': [1, 32, 32],
    'textclassification': [128, ]
}
n_classes = {
    'cifar10': 10,
    'cifar-10-C': 10,
    'har': 6,
    'speechcommand': 10,
    'textclassification': 10
}
model_match = {
    'Reduced_ResNet18': Reduced_ResNet18,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34, 
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet18_pretrained': ResNet18_pretrained,
    'ResNet101_pretrained': ResNet101_pretrained,
    'DCNN_pretrained': DCNN_pretrained, 
    'VGG11_pretrained': VGG11_pretrained,
    'Bert_pretrained': Bert_pretrained
}
transforms_match = {
    'cifar10': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'cifar-10-C': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    'har': transforms.Compose([]),
    'speechcommand': transforms.Compose([]),
    'textclassification': transforms.Compose([])
}

gpu = {
    -1: 'cpu',
    0: 'cuda:0',
    1: 'cuda:1'
}

class dataset_transform(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        if not torch.is_tensor(y): # np.array
            self.y = torch.from_numpy(y).type(torch.LongTensor)
            self.transform = transform
        else: # torch.tensor
            self.y = y
            self.transform = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.transform:
            x = self.transform(self.x[idx])
            if not torch.is_tensor(x):
                x = torch.tensor(x).float()
            else:
                x = x.float()
        else:
            x = self.x[idx]
        return x, self.y[idx]

class dataset_transform_weight(data.Dataset):
    def __init__(self, x, y, w=None, transform=None):
        self.x = x
        if w is None:
            self.w = torch.ones(y.shape[0])
        elif torch.is_tensor(w):
            self.w = w
        else:
            self.w = torch.tensor(w)
        if not torch.is_tensor(y): # np.array
            self.y = torch.from_numpy(y).type(torch.LongTensor)
            self.transform = transform
        else: # torch.tensor
            self.y = y
            self.transform = None

    def __len__(self):
        return len(self.y)#self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        if self.transform:
            x = self.transform(self.x[idx])
            if not torch.is_tensor(x):
                x = torch.tensor(x).float()
            else:
                x = x.float()
        else:
            x = self.x[idx]
        return x, self.y[idx], self.w[idx]

def setup_test_loaders(test_data, args):
    test_loaders = []
    if args.dataset == 'textclassification':
        for (x_test, masks_test, y_test) in test_data:
            test_dataset = data.TensorDataset(torch.tensor(x_test), torch.tensor(masks_test), torch.tensor(y_test))
            test_loader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=True, num_workers=0)
            test_loaders.append(test_loader)
    else:
        for (x_test, y_test) in test_data:
            test_dataset = dataset_transform(x_test, y_test, transform=transforms_match[args.dataset])
            test_loader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=True, num_workers=0)
            test_loaders.append(test_loader)
    return test_loaders

def setup_models(args):
    nclass = n_classes[args.dataset]
    model = model_match[args.model_name](nclass)
    return model

def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim

def setup_scheduler(optimizer, step_size, gamma):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    return scheduler

def record_memory(log_file, device):
    pynvml.nvmlInit()
    gpuDeviceCount = pynvml.nvmlDeviceGetCount() # number of GPUs
    device_id = int(device[-1])
    print('='*100, file=log_file)
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id) #获取GPU i的handle，后续通过handle来处理
    memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)#通过handle获取GPU i的信息
    gpuName = str(pynvml.nvmlDeviceGetName(handle))
    gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
    gpuFanSpeed = pynvml.nvmlDeviceGetFanSpeed(handle)
    gpuPowerState = pynvml.nvmlDeviceGetPowerState(handle)
    gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
    
    print("GPU: %d :"%device_id, "-"*30, file=log_file)
    print("Total Memory:", memoryInfo.total/MEMORY_UNIT, "MB", file=log_file)
    print("Used Memory:", memoryInfo.used/MEMORY_UNIT, "MB", file=log_file)
