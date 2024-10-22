from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils import gpu, AverageMeter, cosine_similarity
from Data.name_match import dataset_transform, dataset_transform_weight, transforms_match, transforms
from torch.utils import data

class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, model, opt, scheduler, args):
        """
        Initialize the ContinualLearner object.
        
        Args:
            model: The neural network model.
            opt: The optimizer.
            scheduler: The learning rate scheduler.
            args: Configuration arguments.
        """
        super(ContinualLearner, self).__init__()
        self.args = args
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.dataset = args.dataset
        self.device = gpu[args.gpu]
        
        self.num_iter = args.num_iter
        self.num_iter_increase = args.num_iter_increase
        self.batch = args.batch
        self.test_batch_num = args.test_batch_num
        self.old_labels, self.new_labels = [], []
        self.task_seen = 0
        self.error_list = []
        self.new_class_score, self.old_class_score = [], []
        self.context_list = args.context_list
        self.enrich_method = args.enrich_method
        self.eval_per_iter = args.eval_per_iter
        
    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def criterion(self, logits, labels):
        """
        Compute the cross-entropy loss.
        
        Args:
            logits: Model outputs.
            labels: True labels.
        
        Returns:
            loss: Computed loss.
        """
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        return ce(logits, labels)
    
    def criterion_each(self, logits, labels):
        """
        Compute the cross-entropy loss for each sample.
        
        Args:
            logits: Model outputs.
            labels: True labels.
        
        Returns:
            loss: Computed loss for each sample.
        """
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='none')
        return ce(logits, labels)
    
    def evaluate(self, test_loaders):
        """
        Evaluate the model on the test set.
        
        Args:
            test_loaders: List of test data loaders.
        
        Returns:
            acc_array: Array of accuracies for each task.
            loss_array: Array of losses for each task.
        """
        self.model.eval()
        acc_array, loss_array = np.zeros(len(test_loaders)), np.zeros(len(test_loaders), dtype=float)
        with torch.no_grad():
            for task, test_loader in enumerate(test_loaders):
                acc, loss = AverageMeter(), AverageMeter()
                if self.dataset == 'textclassification':
                    for i, (batch_x, batch_masks, batch_y) in enumerate(test_loader):
                        batch_x, batch_masks, batch_y = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device)
                        logits = self.model(batch_x, token_type_ids=None, attention_mask=batch_masks, labels=batch_y)[1]
                        _, pred_label = torch.max(logits, 1)
                        loss_tmp = self.criterion(logits, batch_y)
                        acc_tmp = (pred_label == batch_y).sum().item() / batch_y.size(0)
                        acc.update(acc_tmp, batch_y.size(0))
                        loss.update(loss_tmp, batch_y.size(0))
                        if (i + 1) >= self.test_batch_num:
                            break
                else:
                    for i, (batch_x, batch_y) in enumerate(test_loader):
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        logits = self.model.forward(batch_x)
                        _, pred_label = torch.max(logits, 1)
                        loss_tmp = self.criterion(logits, batch_y)
                        acc_tmp = (pred_label == batch_y).sum().item() / batch_y.size(0)
                        acc.update(acc_tmp, batch_y.size(0))
                        loss.update(loss_tmp, batch_y.size(0))
                        if (i + 1) >= self.test_batch_num:
                            break
                acc_array[task] = acc.avg()
                loss_array[task] = loss.avg()
        return acc_array, loss_array