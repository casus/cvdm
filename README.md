<<<<<<< HEAD
# cvdm
=======
# Conditional Variational Diffusion Models

This code implements the Conditional Variational Diffusion Models as described [in the paper](https://arxiv.org/abs/2312.02246).

## Where to get the data?

The datasets that we are using are available online:
- [BioSR](https://github.com/qc17-THU/DL-SR)
- [ImageNet from ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/)
- [HCOCO](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4?tab=readme-ov-file) - only used in model evaluation


## How to prepare environment?

```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
pip install -e .
```

## How to run the code?
1. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs
2. Run the code from the root directory: `python scripts/train.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN`
>>>>>>> 9d9308e (Package CVDM)
