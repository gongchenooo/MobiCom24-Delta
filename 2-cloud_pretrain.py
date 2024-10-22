import torch
import torch.optim as optim
from torch.utils import data as torch_data
import numpy as np
from Models.pretrain import ResNet18_pretrained, ResNet50_pretrained, ResNet101_pretrained
from Models.resnet import Reduced_ResNet18, ResNet18
from Models.HAR_model import DCNN
from Models.speech_model import vgg11
from Data.name_match import n_classes, dataset_transform, gpu
from tqdm import tqdm
import os
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(seed=0)

class Pretrain():
    def __init__(self, root="./", device=0, 
                 model="Reduced_ResNet18", dataset="", 
                 log_root='/Models/Pretrain', 
                 load_path=None):
        
        self.root = root
        self.dataset = dataset
        self.log_root = log_root + "/{}/".format(dataset)
        self.model_name = model
        self.device = gpu[device]
        self.nclass = n_classes[self.dataset]
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.load_data()
        self.load_model(load_path)
        
        
    def load_data(self):
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        if self.dataset == "har":
            self._load_har_data()
        else:
            exit("Wrong dataset!")
        
        print('Train data: {}\tTest data: {}'.format(self.x_train.shape, [i.shape for i in self.x_test]))
        self.train_dataset = dataset_transform(self.x_train, self.y_train)
        self.train_loader = torch_data.DataLoader(self.train_dataset, batch_size=100, shuffle=True, num_workers=0)
        
        self.test_dataset, self.test_loader = [], []        
        for d in range(len(self.x_test)):
            self.test_dataset.append(dataset_transform(self.x_test[d], self.y_test[d]))
            self.test_loader.append(torch_data.DataLoader(self.test_dataset[-1], batch_size=100, shuffle=False))

    def load_model(self, load_path):
        if self.model_name == 'Reduced_ResNet18':
            self.model = Reduced_ResNet18(self.nclass)
        elif self.model_name == 'ResNet18':
            self.model = ResNet18(self.nclass)
        elif self.model_name == 'ResNet18_pretrained':
            self.model = ResNet18_pretrained(self.nclass)
        elif self.model_name == 'DCNN':
            self.model = DCNN()
        elif self.model_name == 'VGG11':
            self.model = vgg11()
        self.model = self.model.to(self.device)
        self.opt = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.opt, 1000, 0.1)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))

    def train(self):
        self.log_file = open(self.log_root+'{}.txt'.format(self.model_name), 'w')
        acc_max = 0.0
        self.model.eval()
        
        for epoch in range(2):
            for i, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.opt.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.opt.step()
                
                if (i + 1) % 250 == 0 or (i + 1) == len(self.train_dataset):
                    test_acc, test_loss = self.evaluate()
                    if test_acc > acc_max:
                        acc_max = test_acc
                        self.save_model(test_acc)
                    print('[{}/{}]\tTest Acc: {:.4f}\tTest Loss: {:.4f}'.format(i, len(self.train_loader), test_acc, test_loss))
                    print('[{}/{}]\tTest Acc: {:.4f}\tTest Loss: {:.4f}'.format(i, len(self.train_loader), test_acc, test_loss), file=self.log_file)
                    self.log_file.flush()
            self.scheduler.step()
    
    def save_model(self, acc):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), f'{self.log_root}{self.model_name}.pth')
        self.model.to(self.device)
        print(f'Save model with acc: {acc:.4f}')
        print(f'Save model with acc: {acc:.4f}', file=self.log_file)
            
    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        return ce(logits, labels)

    def evaluate(self):
        self.model.eval()
        acc_list, loss_list = [], []
        with torch.no_grad():
            for i in range(len(self.datasets)):
                test_loader = self.test_loader[i]
                acc, loss, cnt = 0.0, 0.0, 0
                for j, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.model.forward(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    acc_tmp = (pred_label==batch_y).sum().item()/batch_y.size(0)
                    loss_tmp = self.criterion(logits, batch_y).item()
                    acc += acc_tmp*batch_y.shape[0]
                    loss += loss_tmp*batch_y.shape[0]
                    cnt += batch_y.shape[0]
                    if (j + 1) % 500 == 0 or (j + 1) == len(test_loader):
                        print('{}:{:.4f}'.format(self.datasets[i], acc/cnt))
                        print('{}:{:.4f}'.format(self.datasets[i], acc/cnt), file=self.log_file)
                        acc_list.append(acc/cnt)
                        loss_list.append(loss/cnt)
                        break
        return sum(acc_list)/len(acc_list), sum(loss_list)/len(loss_list)
            
    def _load_har_data(self):
        self.datasets = "hhar", "uci", "motion", "shoaib"
        x_train, y_train, device_train = [], [], []
        for dataset in self.datasets:
            data = np.load(
                self.root+"Data/RawData/har/{}_cloud.npz".format(dataset), allow_pickle=True)
            x_train_tmp, y_train_tmp, device_train_tmp = data["arr_0"], data["arr_1"], data["arr_2"]
            x_train.append(x_train_tmp)
            y_train.append(y_train_tmp)
            device_train.append(device_train_tmp)
        self.x_train, self.y_train, self.device_train = np.concatenate(x_train, 0), np.concatenate(y_train, 0), np.concatenate(device_train, 0)
        
        x_test, y_test, device_test = [], [], []
        for i, dataset in enumerate(self.datasets):
            data = np.load(self.root+"Data/RawData/har/{}_device.npz".format(dataset), allow_pickle=True)
            x_test_tmp, y_test_tmp, device_test_tmp = data["arr_0"], data["arr_1"], data["arr_2"]
            x_test.append(x_test_tmp)
            y_test.append(y_test_tmp)
            device_test.append(device_test_tmp)
        self.x_test, self.y_test, self.device_test = x_test, y_test, device_test
        
model = "DCNN"
dataset = "har"
p = Pretrain(model=model, dataset=dataset)
p.train()