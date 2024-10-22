import torch
from torch.utils import data
from Data.name_match import dataset_transform, transforms_match, transforms
from utils import AverageMeter, cosine_similarity, gpu
from Data.cloud import CloudServer
import numpy as np
import time
import sys

class Delta:
    def __init__(self, args):
        """
        Initialize the Delta object for data enrichment
        
        Args:
            args: Configuration arguments.
        """
        self.cloud_datasets = args.cloud_datasets
        self.enrich_data_num = args.enrich_data_num
        self.enrich_metric = args.enrich_metric
        self.context_list = args.context_list
        self.cloud = CloudServer(args)
        self.dataset = args.dataset
        self.device = gpu[args.gpu]
        self.cluster_topK = min(args.cluster_topK, self.cloud.n_cluster)
        self.temperature = args.enrich_temperature
        self.sampling_gamma = args.sampling_gamma
    
    def data_enrich(self, x_train, y_train, model, enrich_method, task_id, log_file, enrich_cnt_old_dic={}, masks_train=None):
        """
        Enrich the local training data based on the specified enrich method.
        
        Args:
            x_train: Local training data.
            y_train: Local training labels.
            model: Local model.
            enrich_method: Enrichment method ('original', 'random', or 'delta').
            task_id: Current task ID.
            log_file: Log file.
            enrich_cnt_old_dic: Directory weights of old task.
            masks_train: Training masks for text classification.
        
        Returns:
            x_train_enrich: Enriched data.
            masks_train_enrich: Enriched training masks (if applicable).
            y_train_enrich: Enriched labels.
            w_train_enrich: Weights of enriched data for re-weighting in importance sampling.
            enrich_cnt_dic: Directory weights of current task
        """
        
        print('='*20 + ' Data Enrichment Init ' +'='*20)
        
        if enrich_method == "original":
            enrich_cnt_dic = {}
            x_train_enrich, y_train_enrich, w_train_enrich = x_train, y_train, np.ones(x_train.shape[0])
            masks_train_enrich = masks_train if self.dataset == 'textclassification' else None
            
        elif enrich_method == 'random': 
            x_train_enrich, masks_train_enrich, y_train_enrich, w_train_enrich = [], [], [], None
            labels = np.unique(y_train)
            # randomly select data for each label
            cnt = np.random.randint(1, 5, size=labels.shape[0]) 
            enrich_cnt_dic = {labels[i]: round(cnt[i] / cnt.sum() * self.enrich_data_num) for i in range(labels.shape[0])}
            for y in labels:
                idx = np.arange(self.cloud.cluster_pool[y]['x'].shape[0])
                np.random.shuffle(idx)
                idx = idx[:enrich_cnt_dic[y]]
                x_train_enrich.append(self.cloud.cluster_pool[y]['x'][idx])
                y_train_enrich.append(self.cloud.cluster_pool[y]['y'][idx])
                if self.dataset == 'textclassification':
                    masks_train_enrich.append(self.cloud.cluster_pool[y]['masks'][idx])
            
            x_train_enrich, y_train_enrich = np.concatenate(x_train_enrich), np.concatenate(y_train_enrich)
            w_train_enrich = np.ones(x_train_enrich.shape[0])
            if self.dataset == 'textclassification':
                masks_train_enrich = np.concatenate(masks_train_enrich)
            
        elif enrich_method == 'delta':
            sampling_method = "optimal" # random/optimal cloud-side sampling to choose for easy ablation studies
            
            # device-side softmatching to obtain directory weights
            enrich_cnt_dic = self.data_enrich_device_side(x_train, y_train, model, log_file, masks_train)
            
            # cloud-side sampling with specified sampling method
            if self.dataset == 'textclassification':
                x_train_enrich, masks_train_enrich, y_train_enrich, w_train_enrich = self.data_enrich_cloud_side(
                    sampling_method, task_id, enrich_cnt_dic, enrich_cnt_old_dic, log_file)
            else:
                x_train_enrich, y_train_enrich, w_train_enrich = self.data_enrich_cloud_side(
                    sampling_method, task_id, enrich_cnt_dic, enrich_cnt_old_dic, log_file)
            
            for y in enrich_cnt_dic.keys():
                print('label: {}'.format(y), end='\t')
                print('label: {}'.format(y), end='\t', file=log_file)
                enrich_cnt_dic[y] = [i / sum(enrich_cnt_dic[y]) for i in enrich_cnt_dic[y]]
                for i in range(self.cloud.n_cluster):
                    if enrich_cnt_dic[y][i] > 1e-5:
                        print('{}({})'.format(self.cloud.cluster_pool[y]['center_datasets'][i], round(enrich_cnt_dic[y][i], 2)), end='\t')
                        print('{}({})'.format(self.cloud.cluster_pool[y]['center_datasets'][i], round(enrich_cnt_dic[y][i], 2)), end='\t', file=log_file)
                print()
                print(file=log_file)
        
            x_train_enrich, y_train_enrich = np.concatenate(x_train_enrich), np.concatenate(y_train_enrich)
            w_train_enrich = np.concatenate(w_train_enrich)
            if self.dataset == 'textclassification':
                masks_train_enrich = np.concatenate(masks_train_enrich)
        
        print('Enriched Data Shape: ', x_train_enrich.shape)
        print('Enriched Data Shape: ', x_train_enrich.shape, file=log_file)
        print('Weight: ', w_train_enrich, file=log_file)
        print('='*20 + ' Data enrichment Success ' + '='*20)
        
        if self.dataset == 'textclassification':
            return x_train_enrich, masks_train_enrich, y_train_enrich, w_train_enrich, enrich_cnt_dic
        else:
            return x_train_enrich, y_train_enrich, w_train_enrich, enrich_cnt_dic
        
    def data_enrich_device_side(self, x_train, y_train, model, log_file, masks_train=None):
        """
        Enrich data based on the specified methods.
        
        Args:
            x_train: Local data.
            y_train: Local labels.
            model: Local model.
            log_file: Log file.
            masks_train: Local masks for text classification task.
        
        Returns:
            enrich_cnt_dic: directory weights for enrichment
        """
        model.eval()
        labels = np.unique(y_train)
        
        # Initialize feature extractor
        if self.dataset in ['cifar-10-C', 'textclassification']:
            modules = list(model.children())[:-1] # Remove the last fully connected layer
            feature_extractor = torch.nn.Sequential(*modules).to(self.device) 
        elif self.dataset in ['har', 'speechcommand']:
            feature_extractor = model.features    
        
        # Extract features of existing directory dataset
        center_features = {y: [] for y in labels}
        
        for y in labels:
            if self.dataset == 'textclassification':
                center_dataset = data.TensorDataset(torch.tensor(self.cloud.cluster_pool[y]['center_x']),
                                                    torch.tensor(self.cloud.cluster_pool[y]['center_masks']),
                                                    torch.tensor(self.cloud.cluster_pool[y]['center_y']))
                center_loader = data.DataLoader(center_dataset, batch_size=1, shuffle=False, num_workers=0)

                for _, (center_x, center_masks, center_y) in enumerate(center_loader):
                    center_x, center_masks, center_y = center_x.to(self.device), center_masks.to(self.device), center_y.to(self.device)
                    center_feature = model(center_x, token_type_ids=None,
                                            attention_mask=center_masks).hidden_states[-1][:, 0, :].cpu().reshape(1, -1)
                    center_features[y].append(center_feature.clone().detach())
            else:
                center_dataset = dataset_transform(self.cloud.cluster_pool[y]['center_x'], self.cloud.cluster_pool[y]['center_y'],
                                                    transform=transforms_match[self.dataset])
                center_loader = data.DataLoader(center_dataset, batch_size=1, shuffle=False, num_workers=0)

                for _, (center_x, center_y) in enumerate(center_loader):
                    center_x, center_y = center_x.to(self.device), center_y.to(self.device)
                    center_feature = feature_extractor(center_x).cpu().reshape(1, -1)
                    center_features[y].append(center_feature.clone().detach())
                
            center_features[y] = torch.cat(center_features[y])
        print("Directory data's feature shape:", [center_features[y].shape for y in labels])
                
        # Extract features of local new data
        t1 = time.time()
        enrich_cnt_dic = {y: [0 for _ in range(self.cloud.n_cluster)] for y in labels}
        local_features = {y: [] for y in labels}

        if self.dataset == 'textclassification':
            train_dataset = data.TensorDataset(torch.tensor(x_train), torch.tensor(masks_train), torch.tensor(y_train))
            train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

            for _, (batch_x, batch_masks, batch_y) in enumerate(train_loader):
                batch_x, batch_masks, batch_y = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device)
                feature = model(batch_x, token_type_ids=None,
                                attention_mask=batch_masks).hidden_states[-1][:, 0, :].cpu()
                y = int(batch_y[0].item())
                        
                # Device-side soft matching for top K matched clusters
                dis = torch.sum((feature - center_features[y]) ** 2, axis=1)
                _, center_ids = torch.topk(-dis, self.cluster_topK, largest=True, sorted=True)
                dis_softmax = torch.nn.functional.softmax(-dis / self.temperature, dim=0)

                for center_id in center_ids:
                    enrich_cnt_dic[y][center_id] += (self.enrich_data_num / x_train.shape[0] * dis_softmax[center_id].item() / dis_softmax[center_ids].sum().item())

                local_features[y].append(feature.clone().detach())
        else:
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.dataset])
            train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

            for _, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                feature = feature_extractor(batch_x).reshape([1, -1]).cpu()
                y = int(batch_y[0].item())
                # Device-side soft matching for top K matched clusters
                dis = torch.sum((feature - center_features[y]) ** 2, axis=1)
                _, center_ids = torch.topk(-dis, self.cluster_topK, largest=True, sorted=True)
                dis_softmax = torch.nn.functional.softmax(-dis / self.temperature, dim=0)
                        
                for center_id in center_ids:
                    enrich_cnt_dic[y][center_id] += (self.enrich_data_num / x_train.shape[0] * dis_softmax[center_id].item() / dis_softmax[center_ids].sum().item())

                local_features[y].append(feature.clone().detach())

        
        local_features = {y: torch.cat(local_features[y]) for y in labels}
        print("Local data's feature shape", [local_features[y].shape for y in labels])
        t2 = time.time()
        print('Device-Side SoftMatching:\tTotal: {:.3f}ms\tPer Sample:{:.3f}ms'.format((t2 - t1) * 1000, 1000 * (t2 - t1) / len(train_loader)), file=log_file)
        directory_size = sum([sys.getsizeof(self.cloud.cluster_pool[y]['center_x']) for y in labels]) + sum([sys.getsizeof(self.cloud.cluster_pool[y]['center_y']) for y in labels])
        print('Pre-Download_Directory: {:.6f}KB'.format(directory_size / 1024), file=log_file)
        print('Upload_Weight: {:.6f}KB'.format((sys.getsizeof(enrich_cnt_dic)) / 1024), file=log_file)
        
        return enrich_cnt_dic
        
    def data_enrich_cloud_side(self, sample_method, task_id, enrich_cnt_dic, enrich_cnt_old_dic, log_file):
        """
        Perform cloud-side data sampling.
    
        Args:
            sample_method: Cloud-side sampling method ('random' or 'optimal').
            task_id: Current task ID.
            enrich_cnt_dic: Directory weights of current task 
            enrich_cnt_old_dic: Directory weights of past tasks 
            log_file: File to log progress.
        
        Returns:
            x_train_enrich: Enriched training data.
            masks_train_enrich: Enriched training masks (if applicable).
            y_train_enrich: Enriched training labels.
            w_train_enrich: Enriched training weights.
        """
        labels = enrich_cnt_dic.keys()
        
        # Identify center features for each old task, used for optimized cloud-side sampling
        center_feature_old_dic = {}
        for (task_id_old, enrich_cnt_old) in enrich_cnt_old_dic.items():
            center_feature_old = []
            for y_tmp in enrich_cnt_old.keys(): # Identify center for each class
                center_id_tmp = np.argmax(enrich_cnt_old[y_tmp])
                center_feature_old.append(self.cloud.cluster_pool[y_tmp]['center_feature'][center_id_tmp].reshape(1, -1))
            center_feature_old_dic[task_id_old] = torch.cat(center_feature_old) # [class_num, feature_dim]
        
        # cloud-side data sampling
        x_train_enrich, y_train_enrich, w_train_enrich = [], [], []
        masks_train_enrich = [] if self.dataset == 'textclassification' else None
        
        t1 = time.time()
        for y in labels:
            for center_id in range(self.cloud.n_cluster):
                enrich_cnt_dic[y][center_id] = round(enrich_cnt_dic[y][center_id])
                if enrich_cnt_dic[y][center_id] < 1:
                    continue

                idx_y_centerid = np.where(self.cloud.cluster_pool[y]['x_to_center'] == center_id)[0]
                features_y_centerid = self.cloud.cluster_pool[y]['feature'][idx_y_centerid]
                center_feature_new = self.cloud.cluster_pool[y]['center_feature'][center_id]

                enrich_cnt_dic[y][center_id] = min(enrich_cnt_dic[y][center_id], idx_y_centerid.shape[0])
                
                # Randomly select data from each cluster (for ablation studies)
                if sample_method == 'random':
                    idx = np.random.choice(idx_y_centerid, size=enrich_cnt_dic[y][center_id], replace=False)
                    weight = np.ones(enrich_cnt_dic[y][center_id])
                
                # Optimal sampling for each cluster
                elif sample_method == 'optimal':
                    print('Cloud-Side Memory Cost Consists of:', y, center_id, features_y_centerid.shape, center_feature_new.shape, file=log_file)
                    
                    # Importance sampling approach
                    score = (features_y_centerid - center_feature_new.reshape(1, -1)).square().sum(1) # [data_num]
                    score_old = []
                    if task_id == 0:
                        score_old = 0.0
                    else:
                        for task_id_old, center_feature_old in center_feature_old_dic.items():
                            score_old_tmp = []
                            for i in range(len(center_feature_old)):
                                s = (features_y_centerid - center_feature_old[i].reshape(1, -1)).square().sum(1) # [data_num] Sample's distance to old center
                                score_old_tmp.append(s.reshape(1, -1))
                            score_old_tmp = torch.cat(score_old_tmp)  # [center_num, data_num]
                            score_old.append(score_old_tmp.mean(0).reshape(1, -1))  # [1, data_num] Samples' average distance to old task i
                        score_old = torch.cat(score_old)  # [old_task_num, data_num]
                        score_old = score_old.mean(0)  # [data_num] Sample's average distance to all old tasks
                    score = (score + self.sampling_gamma * score_old).sqrt()
                    score /= score.sum()
                    weight = 1 / (score * score.shape[0]) # For re-weight operation in importance sampling for unbiased sampling
                    idx_tmp = torch.multinomial(score, min(enrich_cnt_dic[y][center_id], score.shape[0]), replacement=False).numpy()
                    idx, weight = idx_y_centerid[idx_tmp], weight[idx_tmp]

                x_train_enrich.append(self.cloud.cluster_pool[y]['x'][idx])
                y_train_enrich.append(self.cloud.cluster_pool[y]['y'][idx])
                w_train_enrich.append(weight)
                if self.dataset == 'textclassification':
                    masks_train_enrich.append(self.cloud.cluster_pool[y]['masks'][idx])
        t2 = time.time()
        
        # Report enrichment process information
        print('Cloud-Side Sampling:\t{:.3f}ms'.format((t2 - t1) * 1000), file=log_file)
        print('Download_Enriched_Data: {:.6f}KB'.format((sys.getsizeof(x_train_enrich) + sys.getsizeof(y_train_enrich) + sys.getsizeof(w_train_enrich)) / 1024), file=log_file)
        
        if self.dataset == 'textclassification':
            return x_train_enrich, masks_train_enrich, y_train_enrich, w_train_enrich
        else:
            return x_train_enrich, y_train_enrich, w_train_enrich

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        return ce(logits, labels)
    