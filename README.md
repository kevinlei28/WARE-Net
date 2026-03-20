# WARE-Net: West Africa Road Extraction Network
This repository contains the official Python implementation of WARE-Net. It is a pioneering visual foundation model-based dual-stream architecture designed for fine-grained road surface classification in highly complex remote sensing environments.

The generated first high-resolution road surface cartographic product covering 286 cities across 15 West African countries is available in https://doi.org/10.6084/m9.figshare.31725265.

## 🌟 Key Features

**Foundation Model Encoder:**

Leverages the deep semantic priors of a pre-trained DINOv3 (SAT-493M) backbone for robust feature extraction across diverse topographies. 

**Dual-Stream Decoupled Decoder:**

**Local Texture Decoder:**
Employs multi-scale dilated convolutions (DBlock) for fine-grained material perception, effectively distinguishing dirt roads from bare soil. 
**Global Topology Decoder:**
Integrates a Mamba-based Selective State Space Model with a 2D cross-scan mechanism (SS2D) for long-range structural reasoning and overcoming severe canopy occlusions. 

**Robust Inference & Post-processing:**
Includes our novel Hysteresis Topology Repair (HTR) algorithm and Test-Time Augmentation (TTA) to ensure engineering-grade structural integrity.
<img width="921" height="341" alt="image" src="https://github.com/user-attachments/assets/76a1313b-5057-4d39-8a7d-f0b15bfc50f3" />

Figure 1. The architecture of training phase of WARE-Net.
<img width="735" height="282" alt="image" src="https://github.com/user-attachments/assets/46559834-443a-4447-89b1-fe40d8d36b59" />

Figure 2. The flowchart of the robust inference and post-processing module.



## Result 
<table>
  <tr>
    <td align="center" valign="bottom">
      <img height="300" alt="image" src="https://github.com/user-attachments/assets/e03148fc-40ea-4625-a1f6-922ae48330d9" />
      <br />
      <b>Figure 3. The inference mask in Abidjan, Côte d’Ivoire.</b>
    </td>
    <td align="center" valign="bottom">
      <img height="300" alt="image" src="https://github.com/user-attachments/assets/24680f47-9bfe-4c6d-bcd0-9e19a8cf518e" />
      <br />
      <b>Figure 4. The inference mask in Ho, Ghana.</b>
    </td>
  </tr>
</table>

## 🚀 Getting Started
### 1. Environment Requirements
Our codebase is built upon PyTorch. To ensure absolute compatibility with the visual foundation model encoder, you must first configure the environment according to the official DINOv3 repository.

Please refer to the Official DINOv3 Repository (https://github.com/facebookresearch/dinov3) to install the core dependencies and download the pre-trained weights (SAT-493M).

### 2. Installing Mamba Architecture
After setting up the base environment, you need to install the dependencies for the Global Topology Decoder (Mamba SS2D).

Due to specific CUDA requirements for the Selective State Space Model, please carefully install the following packages:
```bash

pip install causal-conv1d
pip install mamba-ssm
```
(Note: Ensure your CUDA version matches the requirements of mamba-ssm to successfully compile the custom kernels.)

### 3. Configuration
Before running the code, you need to update the local paths to point to your downloaded dataset and pre-trained weights.

Open run.txt in the root directory.

Modify the DATA_DIR and WEIGHTS_PATH variables to match your local environment setup.

### 4. Training and Inference

⚠️ Hardware Notice: All experiments and the default training scripts in this repository were originally conducted on a high-performance computing cluster equipped with 8x NVIDIA A100 Tensor Core GPUs (80 GB memory per node) under Linux operating system.

Multi-GPU Training (Default): The provided scripts are optimized for Distributed Data Parallel (DDP) across 8 GPUs. You can start training directly using:
```bash
bash run.txt
```
Single-GPU Training: If you intend to run the training on a single GPU, you must modify the code to bypass the DDP initialization and adjust the batch size accordingly to prevent Out-Of-Memory (OOM) errors.
