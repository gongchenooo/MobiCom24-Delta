import argparse
from Data.cloud import CloudServer

parser = argparse.ArgumentParser(description="On-Device Continual Learning")
parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')
parser.add_argument('--dataset', dest='dataset', default='cifar-10-C', type=str,
                        choices=['cifar-10-C', 'har', 'speechcommand', 'textclassification'],
                        help='Dataset')
parser.add_argument('--cloud-model', type=str, default='ResNet18_pretrained', 
                        choices=['ResNet18_pretrained',  'DCNN_pretrained', 
                                'VGG11_pretrained', 'Bert_pretrained'],
                        help='cloud model for feature extraction')
parser.add_argument('--gpu', dest='gpu', default=0, type=int,
                        help='-1: cpu, 0: cuda:0, 1: cuda:1')
parser.add_argument('--cluster-method', type=str, default='K-Means', 
                        choices=['Label', 'K-Means', 'DBSCAN', 'GMM', 'MeanShift'],
                        help='data clustering method on the cloud')
parser.add_argument('--cluster-num', type=int, default=5,
                        help='number of clusters (per domain)')
parser.add_argument('--acqu-metric', type=str, default='dis',
                        help='Metrics for data acquisition (choices: dis, cos, dis_qx)')
parser.add_argument('--acqu-value', type=str, default='feature', 
                        choices=['feature', 'grad'],
                        help='Values of data feature')
args = parser.parse_args()

for dataset, cloud_model in [
        # ("cifar-10-C", "ResNet18_pretrained"), 
        # ("har", "DCNN_pretrained"), 
        ("textclassification", "Bert_pretrained")
        ]:
        for cluster_num in [1, 5]:
                args.dataset = dataset
                args.cluster_num = cluster_num
                args.cloud_model = cloud_model
                cloud = CloudServer(args)