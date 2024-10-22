import torch

class Reservior_update(object):
    
    def __init__(self, args):
        super().__init__()

    def update(self, buffer, x, y, task_id, w=None, masks=None, **kwargs):
        w = torch.ones(x.shape[0]) if w is None else w
        
        if task_id not in buffer.task_ids:
            buffer.task_ids.append(task_id)
            buffer.buffer_x[task_id] = x
            buffer.buffer_y[task_id] = y
            buffer.buffer_w[task_id] = w
            buffer.buffer_num[task_id] = y.shape[0]
            if masks is not None:
                buffer.buffer_masks[task_id] = masks        
        else:
            buffer.buffer_x[task_id] = torch.cat([buffer.buffer_x[task_id], x], axis=0)
            buffer.buffer_y[task_id] = torch.cat([buffer.buffer_y[task_id], y], axis=0)
            buffer.buffer_w[task_id] = torch.cat([buffer.buffer_w[task_id], w], axis=0)
            buffer.buffer_num[task_id] += y.shape[0]
            if masks is not None:
                buffer.buffer_masks[task_id] = torch.cat([buffer.buffer_masks[task_id], masks], axis=0)
        assert buffer.buffer_num[task_id] == buffer.buffer_x[task_id].shape[0]
            
        