import numpy as np
import torch

class Random_retrieve(object):
    
    def __init__(self, args):
        super().__init__()
        
    def retrieve(self, buffer, num_retrieve, excl_indices=None, **kwargs):
        """
        Retrieve data from the buffer using a random sampling strategy.

        Args:
            buffer (Buffer): The buffer object containing the data.
            num_retrieve (int): Number of samples to retrieve.
            excl_indices (list, optional): Indices to exclude. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the retrieved data (x, y, w) or (x, masks, y, w) depending on the presence of masks.
        """
        if len(buffer.buffer_masks) == 0:
            x, y, w = [], [], []
            for task_id in buffer.task_ids:
                num_per_task = round(num_retrieve/len(buffer.task_ids))
                x_tmp, y_tmp, w_tmp = self.retrieve_task(buffer, num_per_task, task_id)
                x.append(x_tmp)
                y.append(y_tmp)
                w.append(w_tmp)
            x, y, w = torch.cat(x, axis=0), torch.cat(y, axis=0), torch.cat(w, axis=0)
            return x, y, w
        else:
            x, y, w, masks = [], [], [], []
            for task_id in buffer.task_ids:
                num_per_task = round(num_retrieve/len(buffer.task_ids))
                x_tmp, masks_tmp, y_tmp, w_tmp = self.retrieve_task_with_masks(buffer, num_per_task, task_id)
                x.append(x_tmp)
                masks.append(masks_tmp)
                y.append(y_tmp)
                w.append(w_tmp)
            x, masks, y, w = torch.cat(x, axis=0), torch.cat(masks, axis=0), torch.cat(y, axis=0), torch.cat(w, axis=0)
            return x, masks, y, w
    
    def retrieve_task(self, buffer, num_retrieve, task_id):
        """
        Retrieve data for a specific task from the buffer.

        Args:
            buffer (Buffer): The buffer object containing the data.
            num_retrieve (int): Number of samples to retrieve.
            task_id (int): Task identifier.

        Returns:
            tuple: A tuple containing the retrieved data (x, y, w).
        """
        if task_id not in buffer.task_ids:
            raise ValueError(f'Not buffered data of task {task_id}')
        
        if len(buffer.buffer_masks) == 0:
            if task_id not in buffer.task_ids:
                exit('Not buffered data of task {}'.format(task_id))
            indices = np.arange(buffer.buffer_num[task_id])
            if num_retrieve < indices.shape[0]:
                indices = torch.from_numpy(np.random.choice(indices, num_retrieve, replace=False)).long()
            else:
                pass
            x, y, w = buffer.buffer_x[task_id][indices], buffer.buffer_y[task_id][indices], buffer.buffer_w[task_id][indices]
            return x, y, w
        else:
            return self.retrieve_task_with_masks(buffer, num_retrieve, task_id)
        
        
    
    def retrieve_task_with_masks(self, buffer, num_retrieve, task_id):
        """
        Retrieve data with masks for a specific task from the buffer.

        Args:
            buffer (Buffer): The buffer object containing the data.
            num_retrieve (int): Number of samples to retrieve.
            task_id (int): Task identifier.

        Returns:
            tuple: A tuple containing the retrieved data (x, masks, y, w).
        """
        if task_id not in buffer.task_ids:
            exit('Not buffered data of task {}'.format(task_id))
        if len(buffer.buffer_masks) == 0:
            exit('No data with masks to retrieve')
        indices = np.arange(buffer.buffer_num[task_id])
        if num_retrieve < indices.shape[0]:
            indices = torch.from_numpy(np.random.choice(indices, num_retrieve, replace=False)).long()
        else:
            pass
        x, masks, y, w = buffer.buffer_x[task_id][indices], buffer.buffer_masks[task_id][indices], buffer.buffer_y[task_id][indices], buffer.buffer_w[task_id][indices]
        return x, masks, y, w