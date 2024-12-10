# *(MobiCom'24) Delta: A Cloud-assisted Data Enrichment Framework for On-Device Continual Learning*

Welcome to the repository of Delta, an efficient and effective data enrichment framework designed for on-device continual learning (CL).

The code repository is currently **undergoing updates** as we are re-organizing the code for improved clarity. 

## Table of Contents

1. Repository Structure & Description
2. Requirements
3. Datasets & Preparation
4. Run Commands
5. Acknowledgments and Note

## 1. Repository Structure & Description

    ├──root/Experiments/MobiCom24_Delta     # Root path of the repository
        ├──Agents                           # Files for different continual learning algorithms
            ├──base.py                      # Abstract class for algorithms
            ├──fskd.py                      # Few-shot CL with knowledge distillation
            ├──fspr.py                      # Few-shot CL with parameter freezing
            ├──fsro.py                      # Few-shot CL with robust optimization
            ├──fed_cl.py                    # Federated CL algorithm
            ├──test.py                      # CL with data enrichment algorithms (vanilla, random and delta) 
            ├──delta_class.py               # Implementation of delta operations (device-side softmatching and cloud-side sampling)
        ├──Buffer                           # Files related to buffer management
            ├──buffer.py                    
            ├──name_match.py                # Name-to-function mappings
            ├──random_retrieve.py           # Random retrieval methods
            ├──reservoir_update.py          # Random update methods
        ├──Data                             # Files for create the data stream objects of different datasets
            ├──RawData                      # Raw data files and corresponding preprocessing scripts for each task
                ├──cifar-10-C               # CIFAR-10-C dataset
                ├──har                      # HHAR, UCI, Motion, Shoaib datasets
                    ├──1.preprocess.py      # Preprocessing script
                ├──textclassification       # XGLUE dataset
                    ├──1.preprocess.py      # Preprocessing script
            ├──CloudData                    # Cloud-side data for each task, including public raw data and processed directory dataset
                ├──cifar-10-C               
                ├──har                      
                ├──textclassification       
            ├──cloud.py                     # File for cloud-side operations to generate directory dataset
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
            ├──pretrain.py                  # Methods for loading models with pretrained weights
        ├──2-cloud_pretrain.py              # Cloud-side model pretraining
        ├──3-cloud_preprocess.py            # Cloud-side data processing
        ├──4-main.py                        # On-device continual learning with specified configurations
        ├──5-plot.py                        # Script for plotting the experimental results

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

## 3. Datasets

### 3.1 Downloading Raw Data

All raw data files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/197W-UsmgYEh8Kg5-hmWWxl-nMLqZ1s8j?usp=sharing). Alternatively, you can download each dataset from the following sources:
- **Image Classification**: [CIFAR-10-C](https://github.com/hendrycks/robustness)
- **Human Activity Recognition**: [HHAR](https://dl.acm.org/doi/10.1145/2809695.2809718), [UCI](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930), [Motion](https://dl.acm.org/doi/10.1145/3302505.3310068), [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146)
- **Text CLassification**: [Microsoft XGLUE](https://microsoft.github.io/XGLUE/)
- **Audio Recognition**: [Google Speech Commands](https://arxiv.org/abs/1804.03209)

### 3.2 Preprocessing Raw Data

### 3.3 Preparing Data for Each Context

## 4. Run Commands

1. **Preprocess Raw Data**: Run `1-preprocess.py` for each dataset repository in `Data/RawData/`.
2. **Cloud-side Pretraining**: Execute `2-cloud_pretrain.py` to pretrain models on cloud server. (Note that ResNet and Transformers can directly load pre-trained weights provided by PyTorch)
3. **Cloud-side Data Processing**: Run `3-cloud_preprocess.py` to process public data on the cloud server and generate directory weights/cluster centers.
4. **On-Device Continual Learning**: Run `4-main.py` for each task with specific commands and configurations provided in `Scripts/run_main.sh`, and save results in `Log/`.
5. **Plot Experimental Results**: Run `5-plot.py` to output and visualize the experimental results, and save figures in `Figures/`.

## 5. Acknowledgments and Note

Our code is built upon the repositories of [online-continual-learning](https://github.com/RaptorMai/online-continual-learning) and [Miro](https://github.com/omnia-unist/Miro). We extend our sincere gratitude to their foundational work.

If you have any problems, please feel free to contact us (gongchen@sjtu.edu.cn).
