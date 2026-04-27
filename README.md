# SFC-INRNet
Official PyTorch implementation of "Spatial-Frequency Collaborative Implicit Representation for Infrared Small Target Detection"
# SFC-INRNet: Spatial-Frequency Collaborative Implicit Representation for IRSTD

[![Paper](https://img.shields.io/badge/Paper-The_Visual_Computer-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)

> **Official PyTorch Implementation** of "Spatial-Frequency Collaborative Implicit Representation for Infrared Small Target Detection".
> 
> **Authors:** Zicheng Xu
> 
> **Institution:** School of Mechanical Engineering, Tiangong University, Tianjin, China.

---

## 📢 Abstract
Infrared small target detection (IRSTD) is a critical task in visual surveillance systems, yet challenged by weak target signals and complex background clutter. Existing methods mostly focus on spatial feature learning and neglect frequency-domain information, while discrete upsampling causes detail loss for tiny targets. 

Here we propose **SFC-INRNet**, a spatial-frequency collaborative implicit neural representation framework. The method uses a dual-domain encoder to extract spatial and frequency features, a scale-aware heterogeneous fusion module to integrate cross-domain features, and a continuous implicit decoder to realize high-fidelity small target reconstruction.

## 💡 Introduction
We present the SFC-INRNet framework tailored for the IRSTD task. Experiments on public datasets (NUDT-SIRST, NUAA-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. We propose **SFC-INRNet**, a unified extract-fuse-reconstruct pipeline that bridges the gap between spatial semantics and frequency-domain saliency, leveraging implicit neural representations for high-fidelity detection.
2. An **Adaptive Frequency Enhancement and Reconstruction (AFER)** module is utilized to dynamically mine high-frequency priors and adaptively enhance target signals via explicit frequency band enhancement (EFBE).
3. We devise a **Multi-Scale Heterogeneous Neck (MSHN)** for precise cross-domain alignment and a **Frequency-Aware Implicit Decoder (FAID)** to reconstruct continuous sub-pixel targets without the detail loss of discrete upsampling.

## 🏗️ Network Architecture

<div align="center">
  <img src="overall_structure.png" alt="Overall Architecture" width="850">
  <p align="center">
    <b>Fig. 1. Overall architecture of the proposed SFC-INRNet.</b>
  </p>
</div>

SFC-INRNet comprises three innovative components:
* **AFER Module**: Operates in the frequency domain using 2D FFT to adaptively amplify high-frequency target components.
* **MSHN Neck**: Orchestrates AFGC, LFA, and CGTF modules to fuse multi-scale spatial and frequency features.
* **FAID Decoder**: Utilizes Multi-Frequency Positional Encoding (MFPE) to regress continuous target masks at sub-pixel resolution.

## 🏆 Main Results
Our method achieves state-of-the-art performance across benchmark datasets:

| Dataset | IoU (%) | nIoU (%) | F-measure (%) | Pd (%) | Fa (×10⁻⁶) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NUDT-SIRST** | 94.84 | 95.04 | 96.94 | 99.04 | 2.284 |
| **NUAA-SIRST** | 76.94 | 80.16 | 86.97 | 94.18 | 14.25 |
| **IRSTD-1K** | 65.53 | 65.12 | 79.54 | 92.23 | 32.35 |

*(Refer to Table 1 in our paper for detailed performance comparisons)*

---

## 🚀 Usage Guide

### 1. Environment Preparation
Configure the environment using the provided `requirements.txt`:
```bash
conda create -n sfcinr python=3.8
conda activate sfcinr
pip install -r requirements.txt
