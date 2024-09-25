import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from cvdm.architectures.unet import UNet


def mean_model(input_shape_condition: tf.TensorShape, out_channels: int) -> Model:
    mean_model_input = Input(input_shape_condition)
    unet_output = UNet(
        input_shape_condition, inputs=mean_model_input, out_filters=out_channels
    )
    mean_condition, condition_latents = unet_output
    mean_model_out: Model = Model(mean_model_input, [mean_condition, condition_latents])
    return mean_model_out
