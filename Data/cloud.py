import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
from Data.name_match import dataset_transform, transforms_match, n_classes
from Models.pretrain import ResNet18_pretrained, ResNet50_pretrained, ResNet101_pretrained, DCNN_pretrained, VGG11_pretrained, Bert_pretrained
from utils import gpu, cosine_similarity
from sklearn.cluster import KMeans, DBSCAN, estimate_bandwidth, MeanShift
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
import os
import random

seed = 0
def setup_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed()

class CloudServer:
    def __init__(self, args, root="/root/Experiments/MobiCom24_Delta/Data/RawData"):
        self.root = root
        self.dataset = args.dataset
        self.save_root = "/root/Experiments/MobiCom24_Delta/Data/CloudData/{}/{}/".format(
            self.dataset, args.cloud_model
        )
        # Setup dataset names
        if self.dataset == "cifar-10-C":
            file_names = os.listdir(self.root+"/cifar-10-C")
            self.datasets = [f[:f.find(".")] for f in file_names if "labels" not in f]
        elif self.dataset == "har":
            self.datasets = ["hhar", "uci", "motion", "shoaib"]
        elif self.dataset == "speechcommand":
            self.datasets = ['clear', 'tone', 'noise', 'mixed']
        elif self.dataset == 'textclassification':
            self.datasets = ['de', 'en', 'es', 'fr', 'ru']
            
        self.cloud_model = args.cloud_model
        self.cluster_method = args.cluster_method
        self.device = gpu[args.gpu]
        self.n_cluster = len(self.datasets)*args.cluster_num
        self.enrich_metric, self.enrich_value = args.enrich_metric, args.enrich_value
        
        print("*"*20 + "  Cloud-Side Data Init Begin  " + "*"*20)
        self.download_load()
        self.init_feature_extractor()
        self.init_feature_pool()
        self.init_data_cluster()
        print("*"*20 + "  Cloud-Side Data Init Success  " + "*"*20)
        
    def download_load(self):
        """Download and load the datasets into the data pool."""
        print("1. Load Datasets")
        self.data_pool = {}
        for dataset in self.datasets:
            if self.dataset == "cifar-10-C":
                x, y = np.load(f'{self.root}/cifar-10-C/{dataset}.npy'), np.load(f'{self.root}/cifar-10-C/labels.npy')
                # data with ID from 40000 to 50000 are samples with highest style transformation
                # take last 8000 samples as cloud-side data and first 2000 as device-side dataset
                x, y = x[4 * 10000 + 2000:5 * 10000], y[4 * 10000 + 2000:5 * 10000]
                self.data_pool[dataset] = (x, y)
            elif dataset in ["hhar", "uci", "motion", "shoaib"]:
                data = np.load(f'{self.root}/har/{dataset}_cloud.npz')
                x, y, user = data['arr_0'], data['arr_1'], data['arr_2']
                self.data_pool[dataset] = (x, y)
            elif dataset in ['clear', 'tone', 'noise', 'mixed']:
                data = np.load(f'rawdata/speechcommand/{dataset}_cloud.npy', allow_pickle=True).item()
                x, y = np.expand_dims(np.concatenate([data[u][0] for u in data.keys()]), 1), np.concatenate([data[u][1] for u in data.keys()])
                start = int(len(x) * self.datasets.index(dataset) / len(self.datasets))
                end = int(len(x) * (self.datasets.index(dataset) + 1) / len(self.datasets))
                x, y = x[start:end], y[start:end]
                self.data_pool[dataset] = (x, y)
            elif dataset in ['de', 'en', 'es', 'fr', 'ru']:
                data = np.load(f'{self.root}/textclassification/{dataset}_cloud.npy', allow_pickle=True).item()
                x, masks, y = data['inputs'], data['masks'], data['labels']
                self.data_pool[dataset] = (x, masks, y)
            else:
                exit(f'Download and preprocess the dataset {dataset} first!')
            
    def init_feature_extractor(self):
        """Initialize the feature extractor model."""
        print("2. Load Extractor")
        if self.cloud_model == 'ResNet18_pretrained':
            self.model = ResNet18_pretrained(n_classes[self.dataset])
        elif self.cloud_model == 'ResNet50_pretrained':
            self.model = ResNet50_pretrained(n_classes[self.dataset])
        elif self.cloud_model == 'ResNet101_pretrained':
            self.model = ResNet101_pretrained(n_classes[self.dataset])
        elif self.cloud_model == 'DCNN_pretrained':
            path = "/root/Experiments/MobiCom24_Delta/Models/Pretrain/har/DCNN.pth"
            self.model = DCNN_pretrained(path=path)
        elif self.cloud_model == 'VGG11_pretrained':
            self.model = VGG11_pretrained()
        elif self.cloud_model == 'Bert_pretrained':
            root_path = '/root/Experiments/MobiCom24_Delta/'
            self.model = Bert_pretrained(root_path=root_path)
        else:
            exit('Please define the cloud model first!')
        
        self.model.eval()
        if 'ResNet' in self.cloud_model:
            modules = list(self.model.children())[:-1]
            self.feature_extractor = torch.nn.Sequential(*modules).to(self.device)
        elif 'DCNN' in self.cloud_model:
            self.feature_extractor = self.model.features
        elif 'VGG' in self.cloud_model:
            self.feature_extractor = self.model.features
        elif 'Bert' in self.cloud_model:
            modules = list(self.model.children())[:-2]
            self.feature_extractor = torch.nn.Sequential(*modules).to(self.device)
        self.model.to(self.device)
        
    def init_feature_pool(self):
        """Initialize the feature pool by extracting features from the data pool."""
        print("3. Extract Features")
        self.feature_pool = {dataset: [] for dataset in self.datasets}
        for k in self.datasets:
            if os.path.exists(f'{self.save_root}{k}_datapool.npy'):
                self.feature_pool[k] = np.load(f'{self.save_root}{k}_datapool.npy', allow_pickle=True)
            else:
                if self.dataset == 'textclassification':
                    x, masks, y = self.data_pool[k]
                    idx = np.concatenate([np.where(y == i)[0] for i in np.unique(y)])
                    x, masks, y = x[idx], masks[idx], y[idx]
                    dataset = TensorDataset(torch.tensor(x), torch.tensor(masks), torch.tensor(y))
                    loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
                    for batch_x, batch_masks, batch_y in tqdm(loader):
                        with torch.no_grad():
                            batch_x, batch_masks, batch_y = batch_x.to(self.device), batch_masks.to(self.device), batch_y.to(self.device)
                            batch_feature = self.model(batch_x, token_type_ids=None, attention_mask=batch_masks).hidden_states[-1][:, 0, :]
                            self.feature_pool[k].append(batch_feature.to('cpu'))
                    self.feature_pool[k] = (x, masks, y, torch.cat(self.feature_pool[k]))
                else:
                    x, y = self.data_pool[k]
                    if self.dataset == 'cifar-10-C':
                        idx = np.concatenate([np.random.choice(np.where(y == i)[0], 500, replace=False) 
                                              for i in np.unique(y)] )
                    else:
                        idx = np.concatenate([np.where(y == i)[0] for i in np.unique(y)])
                        
                    x, y = x[idx], y[idx]
                    dataset = dataset_transform(x, y, transforms_match[self.dataset])
                    loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
                    for batch_x, batch_y in tqdm(loader):
                        with torch.no_grad():
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            batch_feature = self.feature_extractor(batch_x).reshape([batch_y.size(0), -1])
                            self.feature_pool[k].append(batch_feature.to('cpu'))
                    self.feature_pool[k] = (x, y, torch.cat(self.feature_pool[k]))
                
                os.makedirs(self.save_root, exist_ok=True)
                np.save(f'{self.save_root}{k}_datapool.npy', self.feature_pool[k])
            print(f'Feature Shape of {k} :\t{self.feature_pool[k][-1].size()}')
            
    def init_data_cluster(self):
        """Initialize the data clustering process."""
        print("4. Data Clustering")
        # Determine the path based on the acquisition metric
        if 'dis' in self.enrich_metric:
            path = f'{self.save_root}/{self.cluster_method}{self.n_cluster}_center_dis.npy'
        elif 'cos' in self.enrich_metric:
            path = f'{self.save_root}{self.cluster_method}{self.n_cluster}_center_cos.npy'
        else:
            exit('Not define such metric!')
            
        # Load the cluster pool if it exists, otherwise initialize it
        if os.path.exists(path):
            self.cluster_pool = np.load(path, allow_pickle=True).item()
        else:
            # Concatenate features and labels from all datasets
            if self.dataset == 'textclassification':
                self.x = np.concatenate([self.feature_pool[k][0] for k in self.datasets])
                self.masks = np.concatenate([self.feature_pool[k][1] for k in self.datasets])
                self.y = np.concatenate([self.feature_pool[k][2] for k in self.datasets])
                self.features = torch.cat([self.feature_pool[k][3] for k in self.datasets])
                self.domain = torch.cat([torch.ones(self.feature_pool[self.datasets[i]][0].shape[0]) * i for i in range(len(self.datasets))])
            else:
                self.x = np.concatenate([self.feature_pool[k][0] for k in self.datasets])
                self.y = np.concatenate([self.feature_pool[k][1] for k in self.datasets])
                self.features = torch.cat([self.feature_pool[k][2] for k in self.datasets])
                self.domain = torch.cat([torch.ones(self.feature_pool[self.datasets[i]][0].shape[0]) * i for i in range(len(self.datasets))])
            print(f'Cloud Feature Shape:\t{self.features.shape}')
            
            # Initialize the clustering model based on the specified method
            if self.cluster_method == 'K-Means':
                cluster_model = KMeans(n_clusters=self.n_cluster, random_state=seed, n_init=20)
            elif self.cluster_method == 'DBSCAN':
                cluster_model = DBSCAN(eps=0.3, min_samples=100)
            elif self.cluster_method == 'GMM':
                cluster_model = GaussianMixture(n_components=self.n_cluster)
            elif self.cluster_method == 'MeanShift':
                bandwidth = estimate_bandwidth(self.features, quantile=0.2, n_samples=300)
                cluster_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            elif self.cluster_method == 'Label':
                cluster_model = None
            else:
                exit('Not define such clustering method!')
            
            # Initialize the cluster pool
            # center_x represents the directory dataset
            if self.dataset == 'textclassification':
                self.cluster_pool = {y: {'x': None, 'y': None, 'masks': None, 'feature': None, 'x_to_center': None, 
                                        'center_x': [], 'center_masks': [], 'center_feature': [], 'center_datasets': []} for y in range(n_classes[self.dataset])}
            else:
                self.cluster_pool = {y: {'x': None, 'y': None, 'feature': None, 'x_to_center': None, 
                                        'center_x': [], 'center_feature': [], 'center_datasets': []} for y in range(n_classes[self.dataset])}
            
            if cluster_model is not None:
                # Perform clustering for each label y
                for y in range(n_classes[self.dataset]):
                    idx = np.where(self.y == y)[0]
                    x_tmp, y_tmp, features_tmp, domain_tmp = self.x[idx], self.y[idx], self.features[idx], self.domain[idx]
                    if self.dataset == 'textclassification':
                        masks_tmp = self.masks[idx]
                    
                    cluster_model.fit(features_tmp)
                    x_to_center_tmp, center_feature_tmp = cluster_model.labels_, torch.tensor(cluster_model.cluster_centers_)
                    
                    self.cluster_pool[y]['x'] = x_tmp
                    self.cluster_pool[y]['y'] = y_tmp
                    self.cluster_pool[y]['feature'] = features_tmp
                    self.cluster_pool[y]['x_to_center'] = x_to_center_tmp
                    self.cluster_pool[y]['center_feature'] = center_feature_tmp
                    if self.dataset == 'textclassification':
                        self.cluster_pool[y]['masks'] = masks_tmp
                    
                    # Find the data centroid for each cluster
                    for i in range(self.n_cluster):
                        dis = torch.sum((features_tmp - center_feature_tmp[i]) ** 2, axis=1)
                        center_x_tmp = np.expand_dims(x_tmp[torch.argmin(dis)], 0)
                        self.cluster_pool[y]['center_x'].append(center_x_tmp)
                        if self.dataset == 'textclassification':
                            center_masks_tmp = np.expand_dims(masks_tmp[torch.argmin(dis)], 0)
                            self.cluster_pool[y]['center_masks'].append(center_masks_tmp)
                        
                        if self.dataset == 'cifar-10-C':
                            idx_i = np.where(x_to_center_tmp == i)[0] // 500  # data_id => domain_id
                            idx_i = domain_tmp[np.where(x_to_center_tmp == i)[0]]
                        else:
                            idx_i = domain_tmp[np.where(x_to_center_tmp == i)[0]]
                        
                        cnt_i = np.array([np.where(idx_i == j)[0].shape[0] for j in range(len(self.datasets))])  # cnt for each domain
                        self.cluster_pool[y]['center_datasets'].append(self.datasets[np.argmax(cnt_i)])
                        
                        print(f'Cluster {i}', np.round(cnt_i / cnt_i.sum(), 2))  # cluster内部各个dataset的占比
                    
                    self.cluster_pool[y]['center_x'] = np.concatenate(self.cluster_pool[y]['center_x'])
                    self.cluster_pool[y]['center_y'] = np.ones(self.cluster_pool[y]['center_x'].shape[0]) * y
                    if self.dataset == 'textclassification':
                        self.cluster_pool[y]['center_masks'] = np.concatenate(self.cluster_pool[y]['center_masks'])
            else:
                pass
            np.save(path, self.cluster_pool)
        
        # Print the shapes of the clustered data
        print('x:', [self.cluster_pool[y]['x'].shape for y in range(n_classes[self.dataset])])
        print('center_x:', [self.cluster_pool[y]['center_x'].shape for y in range(n_classes[self.dataset])])
    
    def comp_var(self, y, center_id):
        """Compute the variance of features in a specific cluster."""
        idx = np.where(self.cluster_pool[y]['x_to_center'] == center_id)[0]
        features = self.cluster_pool[y]['feature'][idx]
        center_feature = self.cluster_pool[y]['center_feature'][center_id]
        var = (features - center_feature.reshape(1, -1)).square().sum(1).mean(0)
        return var