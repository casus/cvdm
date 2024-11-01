# Conditional Variational Diffusion Models

Diffusion models have become popular for their ability to solve complex problems where hidden information needs to be estimated from observed data. Among others, their use is popular in image generation tasks. These models rely on a key hyperparameter of the variance schedule that impacts how well they learn, but recent work shows that allowing the model to automatically learn this hyperparameter can improve both performance and efficiency. Our CVDM package implements Conditional Variational Diffusion Models (CVDM) as described [in the paper](https://arxiv.org/abs/2312.02246) that build on this idea, with the addition of [Zero-Mean Diffusion (ZMD)](https://arxiv.org/pdf/2406.04388), a technique that enhances performance in certain imaging tasks, aiming to make these approaches more accessible to researchers.

## Where to get the data?

The datasets that we are using are available online:
- [BioSR](https://github.com/qc17-THU/DL-SR), the data that we are using has been transformed to .npy files. You can also obtain the data [here](https://drive.google.com/drive/folders/1ZMLAZo4AGX4QASEyd3MGf8LE2B_Bne04?usp=sharing). To generate the training .npz you need to join the parts with 
```cat bio_sr_part* > biosr.npz```
- [ImageNet from ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/)
- [HCOCO](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4?tab=readme-ov-file) - only used in model evaluation

It is assumed that for:
- BioSR super-resolution task, data can be found in the directory specified as dataset_path in configs/biosr.yaml, in two files, x.npy (input) and y.npy (ground truth)
- BioSR phase task, data can be found in the directory specified as dataset_path in configs/biosr_phase.yaml, in one file, y.npy (ground truth). Input to the model will be generated based on the ground truth.
- ImageNet super-resolution task, data can be found in the directory specified as dataset_path in configs/imagenet_sr.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.
- ImageNet phase task, data can be found in the directory specified as dataset_path in configs/imagenet_phase.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.
- HCOCO phase evaluation task, data can be found in the directory specified as dataset_path in configs/hcoco_phase.yaml as a collection of JPEG files. Input to the model will be generated based on the ground truth.

## How to prepare environment?
Create enviroment with
 ```
conda create -n cvdm_env python=3.10
```

Install requirements using
```
pip install -r requirements.txt
```

Note: The Docker image is currently not working. 

## How to run the training code?

1. Download the data or use the sample data available in the data/ directory. The sample data is a fraction of the ImageNet dataset and can be used with configs `imagenet_sr_sample.yaml` or `imagenet_phase_sample.yaml`. You can also use your own data as long as it is in ".npy" format. To do so, use the task type "other".
2. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs. For the description of each parameter, check the documentation in `cvdm/configs/` files.
3. Run the code from the root directory: `python scripts/train.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN`.

`--neptune-token` argument is optional.


## How to run the evaluation code?

1. Download the data. 
1. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs.
2. Run the code from the root directory: `python scripts/eval.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN`.

`--neptune-token` argument is optional.

## How to contribute?

To contribute to the software or seek support, please leave an issue or pull request.

## License
This repository is released under the MIT License (refer to the LICENSE file for details).

