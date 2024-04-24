# Conditional Variational Diffusion Models

This code implements the Conditional Variational Diffusion Models as described [in the paper](https://arxiv.org/abs/2312.02246).

## Where to get the data?

The datasets that we are using are available online:
- [BioSR](https://github.com/qc17-THU/DL-SR), the data that we are using has been transformed to .npy files
- [ImageNet from ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/)
- [HCOCO](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4?tab=readme-ov-file) - only used in model evaluation

It is assumed that for:
- BioSR super-resolution task, data can be found in the directory specified as dataset_path in configs/biosr.yaml, in two files, x.npy (input) and y.npy (ground truth)
- BioSR phase task, data can be found in the directory specified as dataset_path in configs/biosr_phase.yaml, in one file, y.npy (ground truth). Input to the model will be generated based on the ground truth.
- ImageNet super-resolution task, data can be found in the directory specified as dataset_path in configs/imagenet_sr.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.
- ImageNet phase task, data can be found in the directory specified as dataset_path in configs/imagenet_phase.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.
- HCOCO phase evaluation task, data can be found in the directory specified as dataset_path in configs/hcoco_phase_eval.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.

## How to prepare environment?

Run the following code:
```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
pip install -e .
```

## How to run the training code?

1. Download the data. 
1. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs.
2. Run the code from the root directory: `python scripts/train.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN`.

`--neptune-token` argument is optional.

## How to run the training code?

1. Download the data. 
1. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs.
2. Run the code from the root directory: `python scripts/eval.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN`.

`--neptune-token` argument is optional.

## License
This repository is released under the MIT License (refer to the LICENSE file for details).

