import numpy as np
import os
from Data.name_match import setup_test_loaders, setup_models, setup_opt, setup_scheduler, record_memory
from Data.continuum import continuum
from Data.cloud import CloudServer
from Agents.test import Test
from Agents.fskd import FSKD
from Agents.fsro import FSRO
from Agents.fspr import FSPR
from utils import gpu, all_seed

# Mapping of agent names to their respective classes
agent_match = {
    'test': Test,
    'fskd': FSKD,
    'fsro': FSRO,
    'fspr': FSPR
}

def multiple_run(args):
    """
    Execute multiple runs of the experiment.
    """
    if args.store:
        # Construct the log_root directory path based on the scenario
        context_list = '+'.join([d for d in args.context_list.split('+')])
        log_root = os.path.join(
                args.save_path, args.dataset,
                '{}_{}_{}'.format(args.model_name, args.optimizer, args.lr),
                '#Tasks={}_Context={}'.format(args.num_tasks, context_list),
                '{}_{}'.format(args.agent, args.num_per_task)
            )
        
        if args.enrich_method == "delta":
            log_path = log_root + '/{}_{}{}_{}_{}_{}_{}.txt'.format(
                args.enrich_method, args.cluster_method, args.cluster_num,
                args.cluster_topK, args.enrich_temperature,
                args.enrich_data_num, args.seed
            )
        else:
            log_path = log_root + '/{}_{}{}_{}_{}.txt'.format(
                args.enrich_method, args.cluster_method, args.cluster_num,
                args.enrich_data_num, args.seed
            )
        
        # Create the log_root directory if it doesn't exist
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_file = open(log_path, 'w')
        print(args, file=log_file)
    
    # Set the device for GPU usage
    device = gpu[args.gpu]
    
    # Initialize the data continuum
    data_continuum = continuum(args)
    
    # Execute multiple runs
    for run in range(args.num_runs):
        print('=' * 50, file=log_file)
        print(f'Current Run:\t{run + 1}', file=log_file)
        print('=' * 50, file=log_file)
        
        # Set the seed for reproducibility
        all_seed(args.seed + run)
        data_continuum.new_run()
        
        # Set up the model, optimizer, and scheduler
        model = setup_models(args).to(device)
        opt = setup_opt(args.optimizer, model, args.lr, args.wd)
        scheduler = setup_scheduler(opt, args.step_size, args.gamma)
        
        # Set up the test loaders
        test_loaders = setup_test_loaders(data_continuum.test_data(), args)
        
        # Initialize the agent
        agent = agent_match[args.agent](model, opt, scheduler, args)
        
        # Split the context list
        context_list = args.context_list.split('+')
        
        # Train the agent on each task
        if args.dataset == 'textclassification':
            for t, (x_train, masks_train, y_train, labels) in enumerate(data_continuum):
                context = context_list[0] if len(context_list) == 1 else context_list[t]
                print(f'\nTask: {t}\tcontext: {context}\tlabels: {labels}\n', file=log_file)
                # Set the seed for reproducibility
                all_seed(args.seed + run * args.num_tasks + t)
                agent.train_learner(x_train, y_train, data_continuum.test_data(), labels, test_loaders, log_file, masks_train=masks_train)
                log_file.flush()
        else:
            for t, (x_train, y_train, labels) in enumerate(data_continuum):
                context = context_list[0] if len(context_list) == 1 else context_list[t]
                print(f'\nTask: {t}\tcontext: {context}\tlabels: {labels}\n', file=log_file)
                # Set the seed for reproducibility
                all_seed(args.seed + run * args.num_tasks + t)
                agent.train_learner(x_train, y_train, data_continuum.test_data(), labels, test_loaders, log_file)
                log_file.flush()