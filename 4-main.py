import argparse
import numpy as np
import torch
from utils import all_seed, boolean_string
from Experiment.run import multiple_run
# python main.py --num-per-task 50 --num-iter 4000 --optimizer SGD --seed 0 --agent delta --scenario domain --model-name Reduced_ResNet18 --cluster-method K-Means --cloud-model ResNet18_pretrained

def main(args):
    print(args)
    all_seed(args.seed)
    multiple_run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-Device Continual Learning")
    
    # General Settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--dataset', type=str, default='cifar-10-C', choices=['cifar-10-C', 'har', 'speechcommand', 'textclassification'], help='Dataset')
    parser.add_argument('--context-list', type=str, default='fog5+gaussian_noise5+glass_blur5+spatter5+pixelate5', help='On-device contexts in continual learning')
    parser.add_argument('--num-tasks', type=int, default=5, help='Number of tasks')
    parser.add_argument('--num-per-task', type=int, default=10, help='Number of samples per task')
    parser.add_argument('--fixed-order', type=boolean_string, default=False, help='In class incremental, should the class order be fixed (default: %(default)s)')
    
    # Optimizer settings
    parser.add_argument('--gpu', type=int, default=1, help='-1: cpu, 0: cuda:0, 1: cuda:1')
    parser.add_argument('--model-name', type=str, default='ResNet18_pretrained', help='Which model to use')
    parser.add_argument('--agent', type=str, default='test', help='Which CL agent(method) to use')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: %(default)s)')
    parser.add_argument('--num-iter', type=int, default=200, help='The number of iterations used for one task. (default: %(default)s)')
    parser.add_argument('--num-iter-increase', type=float, default=1.0, help='The increasing rate of number of iterations per task. (default: %(default)s)')
    parser.add_argument('--batch', type=int, default=10, help='Batch size (default: %(default)s)')
    parser.add_argument('--test-batch', type=int, default=50, help='Batch size of test data')
    parser.add_argument('--test-batch-num', type=int, default=40, help='Number of batches of test data')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    
    # Scheduler settings
    parser.add_argument('--step-size', type=int, default=1000, help='Adjust the learning rate per ?? rounds')
    parser.add_argument('--gamma', type=float, default=1.0, help='Adjust the learning rate to gamma*current_lr')
    parser.add_argument('--eval-per-iter', type=int, default=20, help='Number of iterations per evaluation')
    
    # Buffer settings
    parser.add_argument('--buffer-size', type=int, default=10000, help='Maximum number of samples stored in buffer')
    parser.add_argument('--update', type=str, default='random', help='Update method')
    parser.add_argument('--retrieve', type=str, default='random', help='Retrieve method')
    
     # Cloud settings
    parser.add_argument('--cloud-datasets', type=str, default=None, help='Datasets used as cloud data pool')
    parser.add_argument('--cloud-model', type=str, default='ResNet18_pretrained', choices=['ResNet18_pretrained', 'DCNN_pretrained', 'VGG11_pretrained', 'Bert_pretrained'], help='Cloud model for feature extraction')
    parser.add_argument('--cloud-load-path', type=str, default='', help='Whether to load model from path')
    parser.add_argument('--cluster-method', type=str, default='K-Means', choices=['Label', 'K-Means', 'DBSCAN', 'GMM', 'MeanShift'], help='Data clustering method on the cloud')
    parser.add_argument('--cluster-num', type=int, default=1, help='Number of clusters (per cloud-side dataset)')
    parser.add_argument('--enrich-data-num', type=int, default=100, help='Number of data samples acquired from the cloud data pool')
    parser.add_argument('--enrich-metric', type=str, default='dis', help='Metrics for data acquisition (choices: dis, cos, dis_qx)')
    parser.add_argument('--enrich-value', type=str, default='feature', choices=['feature', 'grad'], help='Metrics for data clustering')
    parser.add_argument('--enrich-method', type=str, default='original', choices=['original', 'random', 'delta'], help='Data acquisition method')
    parser.add_argument('--enrich-temperature', type=float, default=1.0, help='Temperature for data center matching')
    parser.add_argument('--cluster-topK', type=int, default=2, help='Number of data centers matched for each data sample')
    parser.add_argument('--sampling-gamma', type=float, default=1., help="hyper-parameter to balance new task and past tasks")
    
    ######################## Others ####################################
    parser.add_argument('--store', dest='store', type=boolean_string, default=True,
                        help='Store result or not')
    parser.add_argument('--save-path', dest='save_path', type=str, default='Log',)
    parser.add_argument('--load-path', dest='load_path', type=str, default='')
    args = parser.parse_args()
    
    main(args)