from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Attention, Layer
from tensorflow_addons.layers import GroupNormalization


class AttentionVectorLayer(Layer):
    """
    Building the query, key or value for self-attention
    from the feature map
    """

    def __init__(self, **kwargs) -> None:
        super(AttentionVectorLayer, self).__init__(**kwargs)

    def build(self, input_shape: np.ndarray) -> None:
        self.n_channels = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.n_channels, self.n_channels),
            initializer=VarianceScaling(
                scale=1.0, mode="fan_avg", distribution="uniform"
            ),
            trainable=True,
            name="attention_w",
        )
        self.b = self.add_weight(
            shape=(self.n_channels,),
            initializer="zero",
            trainable=True,
            name="attention_b",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out: tf.Tensor = tf.tensordot(x, self.w, 1) + self.b
        return out

    def get_config(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AttentionVectorLayer:
        return cls(**config)


def attention_block(x: tf.Tensor) -> tf.Tensor:
    """
    Implementing self-attention block, as mentioned in
    https://arxiv.org/pdf/1809.11096.pdf
    """

    x = GroupNormalization(groups=32, axis=-1)(x)

    q = AttentionVectorLayer()(x)
    v = AttentionVectorLayer()(x)
    k = AttentionVectorLayer()(x)

    h: tf.Tensor = Attention()([q, v, k])

    return x + h
