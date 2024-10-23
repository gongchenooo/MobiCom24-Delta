# *(MobiCom'24) Delta: A Cloud-assisted Data Enrichment Framework for On-Device Continual Learning*

Welcome to the official repository of Delta, an efficient and effective data enrichment framework designed for on-device continual learning.

**The repository is still in updating. Sorry for the delay!**

## Table of Contents

1. Repository Structure & Description
2. Requirements
3. Datasets & Preparation
4. Run Commands
5. Acknowledgments and Note

## 1. Repository Structure & Description

    ├──root/Experiments/MobiCom24_Delta     # Root path
        ├──Agents                           # Files for different continual learning algorithms
            ├──base.py                      # Abstract class for algorithms
            ├──fskd.py                      # Few-shot CL with knowledge distillation
            ├──fspr.py                      # Few-shot CL with parameter freezing
            ├──fsro.py                      # Few-shot CL with robust optimization
            ├──test.py                      # CL with data enrichment algorithms (vanilla, random and delta) 
            ├──delta_class.py               # Implementation of delta (device-side softmatching and cloud-side sampling)
        ├──Buffer                           # Files related to buffer management
            ├──buffer.py                    
            ├──name_match.py                # Name-to-function mappings
            ├──random_retrieve.py           # Random retrieval methods
            ├──reservoir_update.py          # Random update methods
        ├──Data                             # Files for create the data stream objects of different datasets
            ├──RawData                      # Raw data files and preprocessing scripts
                ├──cifar-10-C               # CIFAR-10-C dataset
                ├──har                      # HHAR, UCI, Motion, Shoaib datasets
                    ├──1.preprocess.py      # Preprocessing script
                ├──textclassification       # XGLUE dataset
                    ├──1.preprocess.py      # Preprocessing script
            ├──CloudData                    # Cloud-side data, including public data and directory dataset
                ├──cifar-10-C               
                ├──har                      
                ├──textclassification       
            ├──cloud.py                     # File for cloud-side operations for generating directory dataset
            ├──continumm.py                 # Data stream object creation
            ├──name_match.py                # Name-to-function mappings
            ├──utils.py 
        ├──Experiment                       # Files for running specified agents (algorithms) multiple times 
            ├──run.py                       
        ├──Figures                          # Directory for saving figures of experimental results 
        ├──Log                              # Directory for saving final and intermediate results during experiments
        ├──Models                           # Files for backbone models and corresponding pre-training process on cloud server
            ├──Pretrain                     # Directory for saving pre-trained model weights
            ├──HAR_model.py                 # DCNN model for HAR task (requires pretraining)
            ├──resnet.py                    # ResNet model for image tasks (uses pretrained weights from PyTorch)
            ├──speechmodel.py               # VGG model for audio tasks (uses pretrained weights from PyTorch)
            ├──pretrain.py                  # Script for pretraining DNN model for HAR task
        ├──2-cloud_pretrain.py              # Script for cloud-side model pretraining
        ├──3-cloud_preprocess.py            # Script for cloud-side data processing
        ├──4-main.py                        # Script for executing on-device continual learning

## 2. Requirements

Ensure you have the following dependencies installed:

![](https://img.shields.io/badge/python-3.7-green.svg)
![](https://img.shields.io/badge/torch-2.0.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.11.2-blue.svg)
![](https://img.shields.io/badge/scikit--learn-0.24.2-blue.svg)
![](https://img.shields.io/badge/numpy-1.20.3-blue.svg)
![](https://img.shields.io/badge/transformers-4.30.0-blue.svg)
![](https://img.shields.io/badge/tqdm-4.62.3-blue.svg)
![](https://img.shields.io/badge/matplotlib-3.4.3-blue.svg)

## 3. Datasets & Preparation

* All raw data files can be downloaded from [Google Drive](). Alternatively, you can download each dataset from the following sources:
    - **Image Classification**: [CIFAR-10-C](https://github.com/hendrycks/robustness)
    - **Human Activity Recognition**: [HHAR](https://dl.acm.org/doi/10.1145/2809695.2809718), [UCI](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930), [Motion](https://dl.acm.org/doi/10.1145/3302505.3310068), [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146)
    - **Text CLassification**: [Microsoft XGLUE](https://microsoft.github.io/XGLUE/)
    - **Audio Recognition**: [Google Speech Commands](https://arxiv.org/abs/1804.03209)

## 4. Run Commands

1. **Preprocess Raw Data**: Run `1-preprocess.py` for each dataset repository in `Data/RawData/`.
2. **Cloud-side Pretraining**: Execute `2-cloud_pretrain.py` to pretrain models on cloud server. Note that ResNet and Transformers can directly load pre-trained weights provided by PyTorch.
3. **Cloud-side Data Processing**: Run `3-cloud_preprocess.py` to process public data on the cloud server and generate directory weights (cluster centers).
4. **On-Device Continual Learning**: Use `4-main.py` for each task with specific commands and configurations provided in `Scripts/run_main.sh`.
5. **Plot Experimental Results**: Run `5-plot.py` to visualize the experimental results.

## 5. Acknowledgments and Note

Our code is built upon the [online-continual-learning](https://github.com/RaptorMai/online-continual-learning) repository. We extend our sincere gratitude to the [author](https://github.com/RaptorMai) for their foundational work.

If you have any problems, please feel free to contact [us](gongchen@sjtu.edu.cn).
