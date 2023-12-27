# PFSSA — Beyond Pixel-wise Unmixing: Spatial-Spectral Attention Fully Convolutional Networks for Abundance Estimation

This repository contains the official PyTorch implementation of the paper:
Huang J, Zhang P. Beyond Pixel-Wise Unmixing: Spatial–Spectral Attention Fully Convolutional Networks for Abundance Estimation[J]. Remote Sensing, 2023, 15(24): 5694.


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

## Reference

```
@article{huang2023beyond,
  title={Beyond Pixel-Wise Unmixing: Spatial--Spectral Attention Fully Convolutional Networks for Abundance Estimation},
  author={Huang, Jiaxiang and Zhang, Puzhao},
  journal={Remote Sensing},
  volume={15},
  number={24},
  pages={5694},
  year={2023},
  publisher={MDPI}
}
```
