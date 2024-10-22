import torch
from torch.utils import data
from Agents.base import ContinualLearner
from Data.name_match import dataset_transform, transforms_match
from utils import AverageMeter
from Buffer.buffer import Buffer
import numpy as np
from Agents.delta_class import Delta

# python main.py --num-per-task 10 --agent FSRO --num-tasks 5 --scenario class --gpu 1 --num-iter 50 --acqu-metric dis --optimizer SGD  --num-run 3 --num-iter-increase 0.5 --lr 0.01 --batch 10 --context-list fog5+gaussian_noise5+glass_blur5+spatter5+pixelate5
class FSRO(ContinualLearner):
    def __init__(self, model, opt, scheduler, args):
        super(FSRO, self).__init__(model, opt, scheduler, args)
        self.buffer = Buffer(model, args)
        self.buffer_device = Buffer(model, args)
        self.task_id = -1
        self.num_per_task = args.num_per_task        
        self.delta = Delta(args)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.cnt_noise = 5 # used for robust optimization
        
    def pretrain(self, context, method='general', test_loaders=None, log_file=None):
        """
        Pretrain the model on cloud-side data

        Args:
            context (_type_): The context to pretrain on (for customized pretrain)
            method (str): general pretrain or customized pretrain (default 'general').
            test_loaders (list): List of test data loaders.
            log_file (file): File object for logging.
        """
        if method == 'general':
            if self.dataset == 'textclassification':
                x = np.concatenate([self.delta.cloud.feature_pool[d][0] for d in self.delta.cloud.datasets])
                masks = np.concatenate([self.delta.cloud.feature_pool[d][1] for d in self.delta.cloud.datasets])
                y = np.concatenate([self.delta.cloud.feature_pool[d][2] for d in self.delta.cloud.datasets])
            else:
                x = np.concatenate([self.delta.cloud.feature_pool[d][0] for d in self.delta.cloud.datasets])
                y = np.concatenate([self.delta.cloud.feature_pool[d][1] for d in self.delta.cloud.datasets])
        elif method == 'customize':
            x = self.delta.cloud.feature_pool[context][0]
            y = self.delta.cloud.feature_pool[context][1]
        else:
            exit('No such pre-train method')
            
        if self.dataset == 'textclassification':
            train_dataset = data.TensorDataset(torch.tensor(x), torch.tensor(masks), torch.tensor(y))
        else:
            train_dataset = dataset_transform(x, y, transform=transforms_match[self.dataset])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True)
        
        if self.dataset == 'textclassification':
            for i, (batch_x, batch_masks, batch_y) in enumerate(train_loader):
                if i == 0 or (i+1) % 100 == 0:
                    acc_array, loss_array = self.evaluate(test_loaders)
                    print('[Pre-train Iter: {}/{}]\nAcc: {}\tLoss: {}'.format(
                            i, 500, [round(acc_array[i], 3) for i in range(len(acc_array))], [round(loss_array[i], 3) for i in range(len(loss_array))]))
                    print('[Pre-train Iter: {}/{}]\nAcc: {}\tLoss: {}'.format(
                            i, 500, [round(acc_array[i], 3) for i in range(len(acc_array))], [round(loss_array[i], 3) for i in range(len(loss_array))]), file=log_file)
                    log_file.flush()
                self.opt.zero_grad()
                self.model.eval()
                batch_x, batch_masks, batch_y = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x, token_type_ids=None, attention_mask=batch_masks, labels=batch_y)[1]
                loss = self.criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                self.opt.step()
                self.scheduler.step()
                if (i == 499):
                    return
        else:
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if i == 0 or (i + 1) % 100 == 0:
                    acc_array, loss_array = self.evaluate(test_loaders)
                    print('[Pre-train Iter: {}/{}]\nAcc: {}\tLoss: {}'.format(
                            i, 500, [round(acc_array[i], 3) for i in range(len(acc_array))], [round(loss_array[i], 3) for i in range(len(loss_array))]))
                    print('[Pre-train Iter: {}/{}]\nAcc: {}\tLoss: {}'.format(
                            i, 500, [round(acc_array[i], 3) for i in range(len(acc_array))], [round(loss_array[i], 3) for i in range(len(loss_array))]), file=log_file)
                    log_file.flush()
                self.opt.zero_grad()
                self.model.eval()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                if (i == 499):
                    return
                    
    def train_learner(self, x_train, y_train, test_data, labels, test_loaders, log_file, masks_train=None):
        '''
            few-shot CL with robust optimization:
            search flat minima for new tasks
        '''
        """
        Train the learner using robust optimization to searcj flat minima for new tasks.

        Args:
            x_train (np.ndarray): Local data.
            y_train (np.ndarray): Local labels.
            test_data (np.ndarray): Test features.
            labels (list): List of labels.
            test_loaders (list): List of test data loaders.
            log_file (file): File object for logging.
            masks_train (np.ndarray): Attention masks for training data of text classification task.
        """
        # Set up data per task (same number for each label)
        indices = []
        self.task_id += 1
        # Recognize context for customized pre-training for motivating experiments
        if self.dataset == 'cifar-10-C':
            task_context = self.context_list.split('+')[0][:-1]
        elif self.dataset in ['har', 'speechcommand']:
            task_context = self.context_list.split('_')[0]
        elif self.dataset == 'textclassification':
            task_context = self.context_list.split('+')[0]
        else:
            exit(f'Not defined {self.dataset}')
            
        for y in labels:
            idx = np.where(y_train==y)[0]
            np.random.shuffle(idx)
            indices.append(idx[:int(self.num_per_task/len(labels))])
        indices = np.concatenate(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        if self.dataset == 'textclassification':
            masks_train = masks_train[indices]
            train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(masks_train), torch.tensor(y_train))
        else:
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.dataset])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0)
        
        # Pretrain model before the first task/context
        if self.task_id == 0:
            self.pretrain(context=task_context, test_loaders=test_loaders, log_file=log_file)
        
        # Set up trackers
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()
        losses_buffer = AverageMeter()
        acc_buffer = AverageMeter()
    
        # Train model for args.num_iter times
        iter = 0
        for e in range(10000):
            if self.dataset == 'textclassification':
                for _, (batch_x, batch_masks, batch_y) in enumerate(train_loader):
                    if iter == 0 or (iter+1) % self.eval_per_iter == 0:
                        acc_array, loss_array = self.evaluate(test_loaders)
                        self._log_results(iter, acc_array, loss_array, losses_batch, losses_buffer, acc_batch, acc_buffer, log_file)
                    self.opt.zero_grad()
                    self.model.eval()
                    batch_x, batch_masks, batch_y = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device)
                    # Record current parameters 
                    param_values = [p.data.clone() for p in self.params]
                    # Sample noises for robust optimization 
                    for _ in range(self.cnt_noise):
                        noise = [torch.clamp(0.1*torch.randn(p.shape), min=-0.1, max=0.1).to(self.device) for p in param_values]
                        for p, n in zip(self.params, noise):
                            p.data = p.data + n
                        logits = self.model(batch_x, token_type_ids=None, attention_mask=batch_masks, labels=batch_y)[1]
                        loss = self.criterion(logits, batch_y)
                        loss.backward()
                        if self.task_id > 0:
                            buffer_x, buffer_masks, buffer_y, _ = self.buffer.retrieve(num_retrieve=self.batch)
                            buffer_x, buffer_masks, buffer_y = buffer_x.to(self.device), buffer_masks.to(self.device), buffer_y.to(self.device)
                            buffer_logits = self.model(buffer_x, token_type_ids=None, attention_mask=buffer_masks, labels=buffer_y)[1]
                            loss_buffer = self.criterion(buffer_logits, buffer_y)
                            loss_buffer.backward()
                        for p, v in zip(self.params, param_values):
                            p.data = v
                    
                    # Model update
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                    grad = [p.grad.clone() for p in self.params]
                    for p, g in zip(self.params, grad):
                        p.grad.data.copy_(g/self.cnt_noise)
                    self.opt.step()
                    self.scheduler.step()
                    # Store data
                    if e == 0:
                        self.buffer.update(batch_x.cpu(), batch_y.cpu(), self.task_id, masks=batch_masks.cpu())
                    iter += 1
                    if iter == int(self.num_iter * (1.+self.task_id*self.num_iter_increase)):
                        break
            else:
                for _, (batch_x, batch_y) in enumerate(train_loader):
                    if iter == 0 or (iter+1) % self.eval_per_iter == 0:
                        acc_array, loss_array = self.evaluate(test_loaders)
                        self._log_results(iter, acc_array, loss_array, losses_batch, losses_buffer, acc_batch, acc_buffer, log_file)
                    self.opt.zero_grad()
                    self.model.eval()
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    # Record current parameters
                    param_values = [p.data.clone() for p in self.params]
                    # Sample noises for robust optimization
                    for _ in range(self.cnt_noise):
                        noise = [torch.clamp(0.1*torch.randn(p.shape), min=-0.1, max=0.1).to(self.device) for p in param_values]
                        for p, n in zip(self.params, noise):
                            p.data = p.data + n
                        logits = self.model.forward(batch_x)
                        loss = self.criterion(logits, batch_y)
                        loss.backward()
                        if self.task_id > 0:
                            buffer_x, buffer_y, _ = self.buffer.retrieve(num_retrieve=self.batch)
                            buffer_x, buffer_y = buffer_x.to(self.device), buffer_y.to(self.device)
                            buffer_logits = self.model.forward(buffer_x)
                            loss_buffer = self.criterion(buffer_logits, buffer_y)
                            loss_buffer.backward()
                        for p, v in zip(self.params, param_values):
                            p.data = v
                    # model update
                    grad = [p.grad.clone() for p in self.params]
                    for p, g in zip(self.params, grad):
                        p.grad.data.copy_(g/self.cnt_noise)
                    self.opt.step()
                    self.scheduler.step()
                    # store data
                    if e == 0:
                        self.buffer.update(batch_x.cpu(), batch_y.cpu(), self.task_id)
                    iter += 1
                    if iter == int(self.num_iter * (1.+self.task_id*self.num_iter_increase)):
                        break
            if iter == int(self.num_iter * (1.+self.task_id*self.num_iter_increase)):
                    break
    
    def _log_results(self, iter, acc_array, loss_array, losses_batch, losses_buffer, acc_batch, acc_buffer, log_file):
        print('[Iter: {}/{}]\nNew Loss: {:.3f}\tOld Loss: {:.3f}\tNew Acc: {:.2f}\tOld Acc: {:.2f}\nTest Acc: {}\tTest Loss: {}'.format(
                iter, int(self.num_iter*(1+self.num_iter_increase*self.task_id)), 
                losses_batch.avg(), losses_buffer.avg(), acc_batch.avg(), acc_buffer.avg(), 
                [round(acc_array[i], 3) for i in range(len(acc_array))], 
                [round(loss_array[i], 3) for i in range(len(loss_array))]))
        print('[Iter: {}/{}]\nNew Loss: {:.3f}\tOld loss: {:.3f}\tNew Acc: {:.2f}\tOld Acc: {:.2f}\nTest Acc: {}\tTest Loss: {}'.format(
                iter, int(self.num_iter*(1+self.num_iter_increase*self.task_id)), 
                losses_batch.avg(), losses_buffer.avg(), acc_batch.avg(), acc_buffer.avg(), 
                [round(acc_array[i], 3) for i in range(len(acc_array))], 
                [round(loss_array[i], 3) for i in range(len(loss_array))]), file=log_file)
        log_file.flush()