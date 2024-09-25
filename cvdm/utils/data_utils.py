from typing import List

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d


def read_and_patch_image_from_filename(filename: str, im_size: int) -> Image.Image:
    img_orig = cv2.imread(str(filename))
    img_reg = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

    if img_reg.shape[0] < im_size or img_reg.shape[1] < im_size:
        img_patch = Image.fromarray(img_reg)
        img_patch = img_patch.resize((im_size, im_size), Image.LANCZOS)
    else:
        img_patch = extract_patches_2d(img_reg, (im_size, im_size), max_patches=1)[0]
        img_patch = Image.fromarray(img_patch)
    return img_patch


def center_crop(x: np.ndarray, crop_size: int = 2048) -> np.ndarray:
    x_center = x.shape[1] // 2
    y_center = x.shape[0] // 2

    return x[
        y_center - crop_size // 2 : y_center + crop_size // 2,
        x_center - crop_size // 2 : x_center + crop_size // 2,
    ]


def obtain_noisy_sample(x: np.ndarray) -> List[np.ndarray]:
    x_0 = x[0]
    gamma = x[1]

    noise_sample = tf.random.normal(tf.shape(x_0))

    safe_sample = tf.sqrt(gamma) * x_0 + tf.sqrt(1 - gamma) * noise_sample
    safe_mean = tf.sqrt(gamma) * x_0

    return [safe_sample, noise_sample, safe_mean, 1 - gamma]
