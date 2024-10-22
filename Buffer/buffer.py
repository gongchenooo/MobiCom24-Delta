import torch
import numpy as np
from Data.name_match import input_size_match
import Buffer.name_match as name_match

class Buffer(torch.nn.Module):
    def __init__(self, model, args):
        super(Buffer, self).__init__()
        self.args = args
        self.buffer_size = args.buffer_size
        self.input_size = input_size_match[args.dataset]
        self.buffer_x, self.buffer_y, self.buffer_w, self.buffer_num = {}, {}, {}, {}
        self.task_ids = []
        self.buffer_masks = {}

        # Define update and retrieve methods
        self.update_method = name_match.update_methods[args.update](args)
        self.retrieve_method = name_match.retrieve_methods[args.retrieve](args)

    def update(self, x, y, task_id, w=None, masks=None, **kwargs):
        """
        Update the buffer with new data.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Labels.
            task_id (int): Task identifier.
            w (torch.Tensor, optional): Weights. Defaults to None.
            masks (torch.Tensor, optional): Attention masks. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the update method.
        """
        return self.update_method.update(buffer=self, x=x, y=y, w=w, masks=masks, task_id=task_id, **kwargs)
    
    def retrieve(self, num_retrieve, excl_indices=None, **kwargs):
        """
        Retrieve data from the buffer.

        Args:
            num_retrieve (int): Number of samples to retrieve.
            excl_indices (list, optional): Indices to exclude. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the retrieve method.
        """
        return self.retrieve_method.retrieve(buffer=self, num_retrieve=num_retrieve, excl_indices=excl_indices, **kwargs)
    
    def retrieve_task(self, num_retrieve, task_id):
        """
        Retrieve data for a specific task from the buffer.

        Args:
            num_retrieve (int): Number of samples to retrieve.
            task_id (int): Task identifier.

        Returns:
            Any: The result of the retrieve_task method.
        """
        return self.retrieve_method.retrieve_task(buffer=self, num_retrieve=num_retrieve, task_id=task_id)
    
    def discard_task(self, task_id, dis_ratio=0.0):
        """
        Discard a portion of the buffer for a specific task.

        Args:
            task_id (int): Task identifier.
            dis_ratio (float): Ratio of data to discard.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        if dis_ratio < 1e-5 or dis_ratio > 1:
            return False
        
        indices = np.arange(self.buffer_num[task_id])
        new_buffer_num = int(self.buffer_num[task_id] * (1 - dis_ratio))
        
        if new_buffer_num < 1:
            self.task_ids.remove(task_id)
        else:
            indices = torch.from_numpy(np.random.choice(indices, new_buffer_num, replace=False)).long()
            self.buffer_x[task_id] = self.buffer_x[task_id][indices]
            self.buffer_y[task_id] = self.buffer_y[task_id][indices]
            self.buffer_w[task_id] = self.buffer_w[task_id][indices]
            self.buffer_num[task_id] = new_buffer_num
        
        return True
    
    def discard(self, task_id_list=[], dis_ratio_list=[]):
        """
        Discard portions of the buffer for multiple tasks.

        Args:
            task_id_list (list): List of task identifiers.
            dis_ratio_list (list): List of discard ratios corresponding to each task.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        assert len(task_id_list) == len(dis_ratio_list), "Task ID list and discard ratio list must have the same length."
        
        for task_id, dis_ratio in zip(task_id_list, dis_ratio_list):
            self.discard_task(task_id, dis_ratio)
        
        return True
