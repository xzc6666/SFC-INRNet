# SFC-INRNet
Official PyTorch implementation of "Spatial-Frequency Collaborative Implicit Representation for Infrared Small Target Detection"
# Spatial-Frequency Collaborative Implicit Representation for Infrared Small Target Detection

[![Paper](https://img.shields.io/badge/Paper-The_Visual_Computer-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)

> **Official PyTorch Implementation** of SFC-INRNet.
> 
> **Authors:** [Zicheng Xu](https://github.com/xzc6666), Guohua Liu*
> 
> **Institution:** School of Mechanical Engineering, Tiangong University, Tianjin, China.

---

## 📢 Abstract

Infrared small target detection (IRSTD) is a critical task in visual surveillance and search systems, yet challenged by weak target signals and complex background clutter[cite: 206]. Existing methods mostly focus on spatial feature learning and neglect frequency-domain information, while discrete upsampling causes detail loss for tiny targets[cite: 207]. 

Here we propose **SFC-INRNet**, a spatial-frequency collaborative implicit neural representation framework. The method uses a dual-domain encoder to extract spatial and frequency features, a scale-aware heterogeneous fusion module to integrate cross-domain features, and a continuous implicit decoder to realize high-fidelity small target reconstruction[cite: 208, 209].

## 🏗️ Network Architecture

![Overall Architecture](Network_Architecture.png) 
*(Please upload your Fig. 1 here and name it Network_Architecture.png)*

SFC-INRNet operates on a unified extract-fuse-reconstruct pipeline, comprising three core components:
1. **AFER (Adaptive Frequency Enhancement and Reconstruction):** Dynamically mines high-frequency priors alongside spatial extraction.
2. **MSHN (Multi-Scale Heterogeneous Neck):** Utilizes scale-aware aggregation (via AFGC, LFA, and CGTF) for precise cross-domain alignment and semantic refinement.
3. **FAID (Frequency-Aware Implicit Decoder):** Reformulates discrete upsampling as continuous function regression to achieve sub-pixel, high-fidelity geometric reconstruction.

## 🏆 Main Results

[cite_start]Our method achieves state-of-the-art performance across three public benchmark datasets[cite: 210, 413]:

| Dataset | IoU (%) | nIoU (%) | F-measure (%) | Pd (%) | Fa |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NUDT-SIRST** | 94.84 | 95.04 | 96.94 | 99.04 | 2.284 |
| **NUAA-SIRST** | 76.94 | 80.16 | 86.97 | 94.18 | 14.25 |
| **IRSTD-1K** | 65.53 | 65.12 | 79.54 | 92.23 | 32.35 |

[cite_start]*(Refer to Table 1 in our paper for detailed comparisons [cite: 453, 454])*

## 🚀 Getting Started

### 1. Environment Preparation
```bash
conda create -n sfcinr python=3.8
conda activate sfcinr
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
