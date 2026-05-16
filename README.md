# Rethinking Underwater Object Detection Domain Generalization via Frequency Domain Alignment

## Introduction
This repository contains the code for our paper `Rethinking Underwater Object Detection Domain Generalization via Frequency Domain Alignment.`

In real-world underwater environments, affected by complex light absorption and scattering effects, object detection networks often suffer from severe domain shift, which severely hinders the generalization of detectors to unknown target domains. While spatial domain alignment is often employed to mitigate this, such methods are prone to misaligning background noise with foreground targets. To address this challenge, this paper proposes a frequency domain alignment generalization framework termed FDA-Net, which aims to mine robust domain-invariant features via explicit frequency decoupling and aligning. Firstly, a Low-Frequency Mixup (LFM) strategy is designed to generate semantically consistent yet stylistically diverse views, which serve to construct cross-domain sample pairs. Additionally, a Structure-Aware Expert (SAE) module is designed to facilitate feature-level alignment by adaptively filtering domain-specific interference and compensating for missing high-frequency structural details via a soft mixture-of-experts mechanism. Furthermore, a Frequency-Selective Alignment (FSA) loss is proposed to mine domain-invariant features by enforcing strong consistency on high-frequency components and relaxed constraints on low-frequency styles. Extensive experiments demonstrate that FDA-Net achieves state-of-the-art domain generalization on the S-UODAC2020 benchmark, and strikes an optimal accuracy-efficiency tradeoff on the real-world DUO dataset.

<p align="center">
  <img width="5315" height="2190" alt="network" src="https://github.com/user-attachments/assets/f3aed519-9e16-4e8c-9e2a-c28458e75115" />
</p>

## Environment
We tested our two-stage detector (Faster R-CNN) codebase with `PyTorch 1.10.0 + CUDA 11.3 + MMDetection 2.25.1 + MMCV 1.6.0`; one-stage detector (YOLO11s) codebase with `Python 3.10 + Ultralytics 8.3.229`.

## Setup

## Acknowledgements
