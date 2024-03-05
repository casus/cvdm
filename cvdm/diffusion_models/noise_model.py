import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Input

from cvdm.architectures.sr3 import sr3
from cvdm.architectures.unet import UNet


def noise_model(
    input_shape_noisy: tf.TensorShape, out_channels: int, model_type: str
) -> Model:
    noisy_input = Input(input_shape_noisy[:-1] + (out_channels,))
    ref_frame = Input(input_shape_noisy)
    c_inpt = Concatenate()([noisy_input, ref_frame])
    gamma_inp = Input(input_shape_noisy[:-1] + (out_channels,))
    model: Model
    if model_type == "sr3":
        s_model = sr3(
            keras.backend.int_shape(c_inpt)[1:],
            input_shape_noisy[:-1] + (out_channels,),
            out_channels=out_channels,
        )

        noise_out = s_model([c_inpt, gamma_inp])

    else:
        noise_out, _ = UNet(
            input_shape_noisy,
            inputs=c_inpt,
            gamma_inp=gamma_inp,
            out_filters=out_channels,
        )
    model = Model([noisy_input, ref_frame, gamma_inp], noise_out)

    return model
