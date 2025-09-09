# Consistency Distillation for Stable Diffusion 1.5

Implemented as a final project for Yandex School of Data Analysis course on Diffusion Models, this notebook implements **Consistency Distillation** techniques to accelerate Stable Diffusion 1.5 from 50 inference steps down to just 4 steps while maintaining image quality. I generated some fancy pictures at the end of the notebook, please take a look!
<img width="929" height="217" alt="image" src="https://github.com/user-attachments/assets/a1d19233-aa7e-4e9a-b884-a70c471e50ed" />


## Overview

Consistency Models (CM) learn to map any point on a diffusion trajectory directly to the clean data, enabling single-step generation. This implementation covers three approaches:

1. **Consistency Training (CT)** - Training consistency models from scratch
2. **Consistency Distillation (CD)** - Using a pre-trained teacher model 
3. **Multi-boundary CD** - Advanced technique splitting trajectories into segments

## Key Features

- **Fast Generation**: 4-step inference vs. 50-step baseline
- **Memory Efficient**: Uses LoRA adapters
- **Production features**: Mixed precision training, gradient accumulation

## Technical Implementation

### Core Components
- DDIM solver implementation for diffusion sampling
- Noise sampling for forward diffusion process
- Consistency loss functions (MSE and pseudo-Huber)
- Classifier-free guidance integration
- Custom sampling pipeline for consistency models

### Training Optimizations
- **LoRA Adapters**: Efficient fine-tuning with low-rank decomposition
- **Mixed Precision**: FP16/FP32 training for speed and memory
- **Gradient Accumulation**: Effective larger batch sizes
- **Gradient Checkpointing**: Memory optimization

### Model Variants
1. **CT Model**: Self-supervised consistency training
2. **CD Model**: Teacher-student distillation with CFG
3. **Multi-CD Model**: Segmented trajectory approach (4 boundaries)

## Dataset
Uses 5,000 text-image pairs from COCO dataset for efficient training on limited compute (Tesla T4 compatible).

## Results
Progressive quality improvement through the three approaches:
- **Baseline SD1.5**: High quality, 50 steps
- **CT**: Faster but basic quality, 4 steps  
- **CD**: Better quality with teacher guidance, 4 steps
- **Multi-CD**: Best quality with deterministic sampling, 4 steps

## Usage
The notebook provides complete training and inference pipelines, including model uploading to HuggingFace Hub for deployment.

## Requirements
- PyTorch 2.4.1+
- Diffusers 0.30.3
- PEFT 0.8.2
- CUDA-capable GPU (minimum 15GB VRAM)

