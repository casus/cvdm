from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Layer


class ConditionalInstanceNormalization(Layer):
    """
    The goal of conditional instance normalization is to make the
    model aware of the amount of noise to be removed (i.e how many
    steps in the noise diffusion process are being considered). The
    implementation was informed by the appendix in
    https://arxiv.org/pdf/1907.05600.pdf and implementation at
    https://github.com/ermongroup/ncsn/blob/master/models/cond_refinenet_dilated.py
    """

    def __init__(self, **kwargs):
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape: np.ndarray) -> None:
        self.batch_size = input_shape[0][0]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.n_channels = input_shape[0][3]
        self.embedding_dim = input_shape[1][1]
        self.w1 = self.add_weight(
            shape=(self.embedding_dim, self.n_channels),
            initializer=RandomNormal(mean=1.0, stddev=0.02),
            trainable=True,
            name="conditional_w1",
        )
        self.b = self.add_weight(
            shape=(self.embedding_dim, self.n_channels),
            initializer="zero",
            trainable=True,
            name="conditional_b",
        )
        self.w2 = self.add_weight(
            shape=(self.embedding_dim, self.n_channels),
            initializer=RandomNormal(mean=1.0, stddev=0.02),
            trainable=True,
            name="conditional_w2",
        )

    def call(self, inputs: Iterable[tf.Tensor]) -> tf.Tensor:
        x, noise_embedding = inputs
        feature_map_means = tf.math.reduce_mean(x, axis=(1, 2), keepdims=True)
        feature_map_std_dev = tf.math.reduce_std(x, axis=(1, 2), keepdims=True) + 1e-5
        m = tf.math.reduce_mean(feature_map_means, axis=-1, keepdims=True)
        v = tf.math.reduce_std(feature_map_means, axis=-1, keepdims=True) + 1e-5
        gamma = tf.expand_dims(
            tf.expand_dims(tf.tensordot(noise_embedding, self.w1, 1), 1), 1
        )
        beta = tf.expand_dims(
            tf.expand_dims(tf.tensordot(noise_embedding, self.b, 1), 1), 1
        )
        alpha = tf.expand_dims(
            tf.expand_dims(tf.tensordot(noise_embedding, self.w2, 1), 1), 1
        )
        instance_norm = (x - feature_map_means) / feature_map_std_dev
        x = gamma * instance_norm + beta + alpha * (feature_map_means - m) / v
        return x

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ConditionalInstanceNormalization:
        return cls(**config)
