# *(MobiCom'24) Delta: A Cloud-assisted Data Enrichment Framework for On-Device Continual Learning*

Official repository of Delta, an efficient, effective and private data enrichment framework for on-device CL.

## 1. Repo Structure & Decription

    ├──root/Experiments/MobiCom24_Delta     # Root path
        ├──Agents                           # Files for different continual learning algorithms
            ├──base.py                      # Abstract class for algorithms
            ├──fskd.py                      # File for few-shot CL with knowledge distillation
            ├──fspr.py                      # File for few-shot CL with parameter freezing
            ├──fsro.py                      # File for few-shot CL with robust optimization
            ├──test.py                      # File for CL with data enrichment algorithms (vanilla, random and delta) 
            ├──delta_class.py               # Function implementation of delta (device-side softmatching and cloud-side sampling)
        ├──Buffer                           # Files related to buffer (we mainly use random methods)
            ├──buffer.py                    
            ├──name_match.py                
            ├──random_retrieve.py           #File for random retrieval
            ├──reservoir_update.py          #File for random update
        ├──Data                             # Files for create the data stream objects of different datasets
            ├──RawData                      # Raw data files and corresponding data preprocessing files
                ├──cifar-10-C               # CIFAR-10-C
                ├──har                      # HHAR, UCI, Motion, Shoaib
                ├──textclassification       # XGLUE
            ├──CloudData                    # Cloud-side data, including all public data and directory dataset
                ├──cifar-10-C               
                ├──har                      
                ├──textclassification       
            ├──cloud.py                     # File for cloud-side operations for generating directory dataset
            ├──continumm.py                 # Files for creating the data stream objects
            ├──name_match.py                # Mappings from names to functions
            ├──utils.py 
        ├──Experiment                       # File for running the specified agent (algorithm) for multiple times 
            ├──run.py                       
        ├──Figures                          # Directory for saving figures after experiments 
        ├──Log                              # Directory for saving final and intermediate results during experiments
        ├──Models                           # Files for backbone models and corresponding pre-training process on cloud server
            ├──Pretrain                     # Directory for saving the pre-trained model weights
            ├──HAR_model.py                 # Files for DCNN of HAR task (need pretrain)
            ├──resnet.py                    # Files for ResNet of image task (use pretrained model provided by torch)
            ├──speechmodel.py               # Files for VGG models of autio task (use pretrained model provided by torch)
            ├──pretrain.py                  # File for pretraining DNN model of HAR task
        ├──2-cloud_pretrain.py              # File for executing the cloud-side model pretraining
        ├──3-cloud_preprocess.py            # File for executing the cloud-side data processing
        ├──4-main.py                        # File for executing the on-device continual learning

## 2. Requirements
![](https://img.shields.io/badge/python-3.7-green.svg)

![](https://img.shields.io/badge/torch-2.0.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.11.2-blue.svg)
![](https://img.shields.io/badge/scikit--learn-0.24.2-blue.svg)
![](https://img.shields.io/badge/numpy-1.20.3-blue.svg)
![](https://img.shields.io/badge/transformers-4.30.0-blue.svg)
![](https://img.shields.io/badge/tqdm-4.62.3-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.4.3-blue.svg)

## 3. Datasets & Preparation

* All the raw data files can be downloaded from [Google Drive](), including
    - [CIFAR-10-C](https://github.com/hendrycks/robustness) for image classification
    - [HHAR](https://dl.acm.org/doi/10.1145/2809695.2809718), [UCI](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930), [Motion](https://dl.acm.org/doi/10.1145/3302505.3310068), [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146) for human activity recognition task
    - [Microsoft XGLUE](https://microsoft.github.io/XGLUE/) for text classification task
    - [Google Speech Commands](https://arxiv.org/abs/1804.03209) for audio recognition task

You can also download each dataset from the url provided above.

## 4. Run Commands

- Step 1: Run `1-preprocess.py` for each dataset repo in `Data/RawData/`
- Step 2: Run `2-cloud_pretrain.py` to pretrain models on cloud server (ResNet and Transformers can directly load the weights pre-trained  and provided by PyTorch)
- Step 3: Run `3-cloud_preprocess.py` to process the public data on cloud server and generate directory weights (i.e. cluster centers)
- Step 4: Run `4-main.py` for each task with specific commands and configurations provided in `Scripts/run_main.sh`
- Step 5: Run `5-plot.py` to plot the experimental results.

## 5. Acknowledgments and Note

Our code is implemented based on [online-continual-learning](https://github.com/RaptorMai/online-continual-learning) and we sincerely thank the [author](https://github.com/RaptorMai).
