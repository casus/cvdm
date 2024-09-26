#!/bin/bash

if ! command -v micromamba &> /dev/null
then
    echo "micromamba could not be found. Please install it first."
    exit
fi 

eval "$(micromamba shell hook --shell )"

ENV_NAME="test_2"
micromamba create -n $ENV_NAME -y
micromamba activate $ENV_NAME

echo "Micromamba environment '$ENV_NAME' activated."

micromamba install -n $ENV_NAME \
    tensorflow-gpu==2.15.0 \
    keras==2.15.* \
    matplotlib==3.8.0 \
    tqdm==4.65.0 \
    scikit-learn==1.4.2 \
    scikit-image==0.22.0 \
    einops==0.7.0 \
    neptune==1.10.2 -y

echo "Micromamba dependencies installed."

pip3 install opencv-python==4.9.0.80
pip3 install tensorflow-addons==0.23.0
pip3 install cupy-cuda12x==13.3.0


python -m pip install .

echo "Pip packages installed."