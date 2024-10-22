from Data.data_objects import data_objects

class continuum(object):
    def __init__(self, args):
        self.data_object = data_objects[args.dataset](args)
        self.num_tasks = args.num_tasks
        self.cur_task = 0
        self.cur_run = -1
        self.dataset = args.dataset

    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Get the next task in the continuum.
        
        Returns:
            x_train: Training data.
            masks_train: Training masks (if applicable).
            y_train: Training labels.
            labels: Unique labels in the training data.
        
        Raises:
            StopIteration: If all tasks have been processed.
        """
        if self.cur_task == self.num_tasks:
            raise StopIteration
        
        if self.dataset == 'textclassification':
            x_train, masks_train, y_train, labels = self.data_object.new_task(self.cur_task)
        else:
            x_train, y_train, labels = self.data_object.new_task(self.cur_task)
            masks_train = None  # Initialize masks_train to None for non-textclassification datasets
        
        self.cur_task += 1
        
        if masks_train is not None:
            return x_train, masks_train, y_train, labels
        else:
            return x_train, y_train, labels
    
    def new_run(self):
        self.cur_task = 0
        self.cur_run += 1
        self.data_object.new_run()

    def test_data(self):
        return self.data_object.get_test_set()
    
    

    