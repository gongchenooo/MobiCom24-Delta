import torch
from torch.utils import data
from Agents.base import ContinualLearner
from Data.name_match import dataset_transform, dataset_transform_weight, transforms_match, record_memory
from utils import AverageMeter
from Buffer.buffer import Buffer
import numpy as np
from Agents.delta_class import Delta

class Test(ContinualLearner):
    def __init__(self, model, opt, scheduler, args):
        super(Test, self).__init__(model, opt, scheduler, args)
        self.buffer = Buffer(model, args)
        self.buffer_device = Buffer(model, args)
        self.task_id = -1
        self.num_per_task = args.num_per_task
        self.delta = Delta(args)
        self.enrich_cnt_dic = {}
        self.enrich_data_num = args.enrich_data_num

    def train_learner(self, x_train, y_train, test_data, labels, test_loaders, log_file, masks_train=None):
        """
        Train the learner on the given data.
        
        Args:
            x_train: Local training data.
            y_train: Local training labels.
            test_data: Test data.
            labels: Unique labels.
            test_loaders: Data loaders for testing.
            log_file: Log file.
            masks_train: Masks for text classification task.
        """
        
        # Set up data per task (same number for each label)
        indices, context_list = [], self.context_list.split('+')
        self.task_id += 1
        task_context = context_list[0][:-1] if len(context_list) == 1 else context_list[self.task_id][:-1]
        for y in labels:
            idx = np.where(y_train==y)[0]
            np.random.shuffle(idx)
            indices.append(idx[:int(self.num_per_task / len(labels))])
        indices = np.concatenate(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        if self.dataset == 'textclassification':
            masks_train = masks_train[indices]
        
        # Data enrichment for new task(context)
        if self.dataset == 'textclassification':
            x_train_enrich, masks_train_enrich, y_train_enrich, w_train_enrich, enrich_cnt_dic = self.delta.data_enrich(
                x_train=x_train, y_train=y_train, masks_train=masks_train,
                model=self.model, enrich_method=self.enrich_method, task_id=self.task_id, 
                log_file=log_file, enrich_cnt_old_dic=self.enrich_cnt_dic)
        else:
            x_train_enrich, y_train_enrich, w_train_enrich, enrich_cnt_dic = self.delta.data_enrich(
                x_train=x_train, y_train=y_train, 
                model=self.model, enrich_method=self.enrich_method, task_id=self.task_id, 
                log_file=log_file, enrich_cnt_old_dic=self.enrich_cnt_dic)
        self.enrich_cnt_dic[self.task_id] = enrich_cnt_dic # record enrichment result for old task
        
        # Prepare training data loader
        if self.dataset == 'textclassification':
            train_dataset = data.TensorDataset(torch.tensor(x_train_enrich), torch.tensor(masks_train_enrich), 
                                               torch.tensor(y_train_enrich), torch.tensor(w_train_enrich))
        else:
            train_dataset = dataset_transform_weight(x_train_enrich, y_train_enrich, w_train_enrich, 
                                                     transform=transforms_match[self.dataset])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0)
        
        # Set up tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()
        losses_buffer = AverageMeter()
        acc_buffer = AverageMeter()
        
        # Train model for args.num_iter times
        iter = 0
        for e in range(10000):
            if self.dataset == 'textclassification':
                for _, (batch_x, batch_masks, batch_y, batch_w) in enumerate(train_loader):
                    if iter == 0 or (iter + 1) % self.eval_per_iter == 0:
                        acc_array, loss_array = self.evaluate(test_loaders)
                        self._log_results(iter, acc_array, loss_array, losses_batch, losses_buffer, acc_batch, acc_buffer, log_file)
                        
                    # Model update of new data batch
                    self.model.eval()
                    self.opt.zero_grad()
                    batch_x, batch_masks, batch_y, batch_w = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device), batch_w.to(self.device)
                    logits = self.model(batch_x, token_type_ids=None, attention_mask=batch_masks, labels=batch_y)[1]
                    loss = (self.criterion_each(logits, batch_y) * batch_w).mean()
                    loss.backward()
                    
                    # Update tracker
                    _, pred_label = torch.max(logits, 1)
                    acc_tmp = (pred_label==batch_y).sum().item() / batch_y.size(0)
                    acc_batch.update(acc_tmp, batch_y.size(0))
                    losses_batch.update(loss.item(), batch_y.size(0))
                    
                    # Model update of old data batch
                    if self.task_id > 0:
                        buffer_x, buffer_masks, buffer_y, buffer_w = self.buffer.retrieve(num_retrieve=self.batch)
                        buffer_x, buffer_masks, buffer_y, buffer_w = buffer_x.to(self.device), buffer_masks.to(self.device), buffer_y.to(self.device), buffer_w.to(self.device)
                        buffer_logits = self.model(buffer_x, token_type_ids=None, attention_mask=buffer_masks, labels=buffer_y)[1]
                        loss_buffer = (self.criterion_each(buffer_logits, buffer_y) * buffer_w).mean()
                        _, pred_label = torch.max(buffer_logits, 1)
                        acc_tmp = (pred_label==buffer_y).sum().item() / buffer_y.size(0)
                        acc_buffer.update(acc_tmp, buffer_y.size(0))
                        losses_buffer.update(loss_buffer.item(), buffer_y.size(0))
                        loss_buffer.backward()
                    
                    # Update model
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
                    self.opt.step()
                    self.scheduler.step()
                    if e == 0: # store the current task's data in buffer
                        self.buffer.update(batch_x.cpu(), batch_y.cpu(), self.task_id, w=batch_w.cpu(), masks=batch_masks.cpu())
                    iter += 1
                    if iter == int(self.num_iter * (1.+self.task_id*self.num_iter_increase)):
                        break
                if iter == int(self.num_iter * (1.+self.task_id*self.num_iter_increase)):
                        break
                
            else:
                for _, (batch_x, batch_y, batch_w) in enumerate(train_loader):
                    if iter == 0 or (iter+1) % self.eval_per_iter == 0:
                        acc_array, loss_array = self.evaluate(test_loaders)
                        self._log_results(iter, acc_array, loss_array, losses_batch, losses_buffer, acc_batch, acc_buffer, log_file)
                        
                    # Model update of new data batch
                    self.model.eval()
                    self.opt.zero_grad()
                    batch_x, batch_y, batch_w = batch_x.to(self.device), batch_y.to(self.device), batch_w.to(self.device)
                    logits = self.model.forward(batch_x)
                    loss = (self.criterion_each(logits, batch_y) * batch_w).mean()
                    loss.backward()
                    
                    # Update tracker
                    _, pred_label = torch.max(logits, 1)
                    acc_tmp = (pred_label==batch_y).sum().item() / batch_y.size(0)
                    acc_batch.update(acc_tmp, batch_y.size(0))
                    losses_batch.update(loss.item(), batch_y.size(0))
                    
                    # Model update of old data batch
                    if self.task_id > 0:
                        buffer_x, buffer_y, buffer_w = self.buffer.retrieve(num_retrieve=self.batch)
                        buffer_x, buffer_y, buffer_w = buffer_x.to(self.device), buffer_y.to(self.device), buffer_w.to(self.device)
                        buffer_logits = self.model.forward(buffer_x)
                        loss_buffer = (self.criterion_each(buffer_logits, buffer_y) * buffer_w).mean()
                        _, pred_label = torch.max(buffer_logits, 1)
                        acc_tmp = (pred_label==buffer_y).sum().item() / buffer_y.size(0)
                        acc_buffer.update(acc_tmp, buffer_y.size(0))
                        losses_buffer.update(loss_buffer.item(), buffer_y.size(0))
                        loss_buffer.backward()
                        
                    # Update model
                    self.opt.step()
                    self.scheduler.step()
                    if e == 0:
                        self.buffer.update(batch_x.cpu(), batch_y.cpu(), self.task_id, w=batch_w.cpu())
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