# PFSSA — Beyond Pixel-wise Unmixing: Spatial-Spectral Attention Fully Convolutional Networks for Abundance Estimation

This repository contains the official PyTorch implementation of the paper:
Jiaxiang Huang, Puzhao Zhang. Beyond Pixel-wise Unmixing: Spatial-Spectral Attention Fully Convolutional Networks for Abundance Estimation

## Reproducing Results

### Installation for Reproducibility

For ease of reproducibility, we suggest you install `Anaconda` before executing the following commands.

```bash
git clone https://github.com/JiaxiangHuang/PFSSA.git
cd PFSSA
mkdir checkpoints
cd src
conda create -n pfssa python=3.7
conda activate pfssa
pip install -r ./requirements.txt 
sh run.sh
```
