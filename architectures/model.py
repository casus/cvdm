import tensorflow as tf
from keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization
import tensorflow_addons as tfa
import architectures.ddpm_blocks as components


def tile_gamma(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    gamma = x[0]
    noisy_y = x[1]
    return tf.ones_like(noisy_y) * tf.expand_dims(tf.expand_dims(gamma, axis=-1), axis=-1)


def time_network(t_shape, n_hidden, act='softplus', mult=True):
    t_input = Input(t_shape)
    l1 = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_constraint=tf.keras.constraints.NonNeg())(t_input)
    x = Conv2D(n_hidden, kernel_size=(3, 3), padding='same', activation='sigmoid',
               kernel_constraint=tf.keras.constraints.NonNeg())(l1)
    x = Conv2D(1, kernel_size=(1, 1), padding='same', kernel_constraint=tf.keras.constraints.NonNeg())(x)
    if mult:
        out = Activation(act)(x + l1) * t_input
    else:
        out = Activation(act)(x + l1)
    model = Model(t_input, out)
    return model


def UNet_ddpm(
        shape,
        previous_tensor=None,
        inputs=None,
        gamma_inp=None,
        full_model=False,
        noise_input=False,
        z_enc=False,
        out_filters=2,
        positive_w=False,
        base_filters=32):
    if positive_w:
        p_constraint = tf.keras.constraints.non_neg()
    else:
        p_constraint = None

    if inputs is None:
        inputs = Input(shape)
    if gamma_inp is None:
        gamma_inp = Input((1,))
        tiled_gamma = Lambda(tile_gamma)([gamma_inp, inputs])
    else:
        tiled_gamma = gamma_inp

    if noise_input:
        noise_in = Input((128,))
        # c_noise = Concatenate()([noise_in, noise_in])
        # c_inpt = Concatenate()([inputs, noise_in])
        conv1 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(inputs)
    else:
        conv1 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(inputs)

    conv_gamma = Conv2D(base_filters, 7, activation='softplus', padding='same', kernel_constraint=p_constraint)(
        tiled_gamma)
    conv1 = Add()([conv_gamma, conv1])
    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = (Conv2D(base_filters * 2, 3, activation='softplus', padding='same', kernel_constraint=p_constraint))(pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv2D(base_filters * 2, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(base_filters * 4, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv2D(base_filters * 4, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(base_filters * 8, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv2D(base_filters * 8, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Attention(512)(pool4)

    conv5 = Conv2D(base_filters * 16, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv2D(base_filters * 16, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv5)
    conv5 = InstanceNormalization()(conv5)

    print(K.int_shape(conv5))
    if previous_tensor is not None:
        previous_tensor = Conv2DTranspose(1024, 3, strides=2, activation='softplus', padding='same')(previous_tensor)
        previous_tensor = Attention(1024)(previous_tensor)
        conv5 = Add()([conv5, previous_tensor])

    if z_enc is not False:
        gamma_vec = Concatenate()([z_enc, conv5])
    else:
        gamma_vec = conv5

    up6 = Conv2D(
        base_filters * 16,
        2,
        activation='softplus',
        padding='same')(
        Conv2DTranspose(512, 3, activation='softplus', padding='same', strides=2, kernel_constraint=p_constraint)(
            gamma_vec))

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(base_filters * 8, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(merge6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv2D(base_filters * 8, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv6)
    conv6 = InstanceNormalization()(conv6)
    # conv6 = Attention(512)(conv6)
    up7 = Conv2D(
        base_filters * 4,
        2,
        activation='softplus',
        padding='same')(
        Conv2DTranspose(256, 3, activation='softplus', padding='same', strides=2, kernel_constraint=p_constraint)(
            conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(base_filters * 4, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(merge7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv2D(base_filters * 4, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv7)
    conv7 = InstanceNormalization()(conv7)
    # conv7 = Attention(256)(conv7)
    up8 = Conv2D(
        base_filters * 2,
        2,
        activation='softplus',
        padding='same')(
        Conv2DTranspose(128, 3, activation='softplus', padding='same', strides=2, kernel_constraint=p_constraint)(
            conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(base_filters * 2, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(merge8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv2D(base_filters * 2, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = Conv2D(
        base_filters,
        2,
        activation='softplus',
        padding='same')(
        Conv2DTranspose(64, 3, activation='softplus', padding='same', strides=2, kernel_constraint=p_constraint)(conv8))

    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(merge9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(base_filters, 3, activation='softplus', padding='same', kernel_constraint=p_constraint)(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv10 = Conv2D(out_filters, 1, activation='linear', padding='same', kernel_constraint=p_constraint)(conv9)
    if full_model and noise_input:
        out = Add()([inputs, conv10])
        model = Model([noise_in, inputs], out)
        model.summary()
        return model
    elif full_model:
        model = Model([inputs, gamma_inp], conv10)
        return model
    return conv10, conv5


def sr3_model(
        img_shape,
        gamma_shape,
        channel_dim: int = 64,
        channel_ramp_multiplier=(1, 2, 4, 8, 8),  # 1, 2, 4, 8, 8
        attention_resolution=(8,),  # 8
        out_channels: int = 3,
        num_resblock: int = 3,  # 3
        dropout: float = 0,
        use_deep_blocks: bool = False,
        resample_with_conv: bool = False) -> tf.keras.Model:
    if use_deep_blocks:
        upblock = components.up_deep_resblock
        block = components.deep_resblock
    else:
        upblock = components.up_resblock
        block = components.resblock

    num_resolutions = len(channel_ramp_multiplier)
    combined_images = Input(img_shape)
    gamma_inp = Input(gamma_shape)
    noise_level_embedding = Conv2D(channel_dim, 7, activation='softplus', padding='same')(
        gamma_inp)

    x = combined_images
    x = tf.keras.layers.Conv2D(channel_dim, 3, padding='same')(x)
    skip_connections = [x]
    for i, multiplier in enumerate(channel_ramp_multiplier):
        for j in range(num_resblock):
            noise_level_embedding = Conv2D(channel_dim * multiplier, 1, padding='same')(noise_level_embedding)
            x = block(x, noise_level_embedding, channel_dim * multiplier, dropout=dropout)
            if x.shape[1] in attention_resolution:
                x = components.attention_block(x)
            skip_connections.append(x)
        if i != num_resolutions - 1:
            x = components.downsample(x, use_conv=resample_with_conv)
            noise_level_embedding = components.downsample(noise_level_embedding, use_conv=resample_with_conv)
            skip_connections.append(x)

    x = block(x, noise_level_embedding, dropout=dropout)

    x = components.attention_block(x)

    x = block(x, noise_level_embedding, dropout=dropout)

    for i, multiplier in reversed(list(enumerate(channel_ramp_multiplier))):
        for j in range(num_resblock + 1):

            noise_level_embedding = Conv2D(channel_dim * multiplier, 1, padding='same')(noise_level_embedding)
            x = upblock(x, skip_connections.pop(), noise_level_embedding, channel_dim * multiplier, dropout=dropout)

            if x.shape[1] in attention_resolution:
                x = components.attention_block(x)
        if i != 0:
            x = components.upsample(x, use_conv=resample_with_conv)
            noise_level_embedding = components.upsample(noise_level_embedding, use_conv=resample_with_conv)

    assert (len(skip_connections) == 0)

    x = tfa.layers.GroupNormalization(groups=32, axis=-1)(x)
    x = tf.keras.activations.swish(x)
    outputs = tf.keras.layers.Conv2D(out_channels, 3, padding='same')(x)

    model = tf.keras.Model([combined_images, gamma_inp], outputs)
    return model


def UNet(shape, out_channels=3, inputs=None, full_model=False, z_enc=None):
    if inputs is None:
        inputs = Input(shape)
    conv1 = Conv2D(64, 3, activation='softplus', padding='same')(inputs)
    conv1 = InstanceNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='softplus', padding='same')(conv1)
    conv1 = InstanceNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='softplus', padding='same')(pool1)
    conv2 = InstanceNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='softplus', padding='same')(conv2)
    conv2 = InstanceNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='softplus', padding='same')(pool2)
    conv3 = InstanceNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='softplus', padding='same')(conv3)
    conv3 = InstanceNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='softplus', padding='same')(pool3)
    conv4 = InstanceNormalization()(conv4)
    conv4 = Conv2D(1024, 3, activation='softplus', padding='same')(conv4)
    conv4 = InstanceNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Attention(1024)(pool4)
    conv5 = Conv2D(1024, 3, activation='softplus', padding='same')(pool4)
    conv5 = InstanceNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='softplus', padding='same')(conv5)
    if z_enc is None:
        cur_hidden = InstanceNormalization()(conv5)
    else:
        # ch_conv = K.int_shape(conv5)[-1]
        # ch_z = K.int_shape(z_enc)[-1]
        # tiled_z = Concatenate()([z_enc] * (ch_conv // ch_z))

        cur_hidden = InstanceNormalization()(conv5) + z_enc

    up6 = Conv2D(
        512,
        2,
        activation='softplus',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(cur_hidden))

    merge6 = Concatenate()([conv4, up6])
    conv6 = Conv2D(512, 3, activation='softplus', padding='same')(merge6)
    conv6 = InstanceNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='softplus', padding='same')(conv6)
    conv6 = InstanceNormalization()(conv6)
    # conv6 = Attention(512)(conv6)
    up7 = Conv2D(
        256,
        2,
        activation='softplus',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='softplus', padding='same')(merge7)
    conv7 = InstanceNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='softplus', padding='same')(conv7)
    conv7 = InstanceNormalization()(conv7)
    # conv7 = Attention(256)(conv7)
    up8 = Conv2D(
        128,
        2,
        activation='softplus',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='softplus', padding='same')(merge8)
    conv8 = InstanceNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='softplus', padding='same')(conv8)
    conv8 = InstanceNormalization()(conv8)

    up9 = Conv2D(
        64,
        2,
        activation='softplus',
        padding='same')(
        UpSampling2D(
            size=(
                2,
                2))(conv8))

    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='softplus', padding='same')(merge9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='softplus', padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation='softplus', padding='same')(conv9)
    conv9 = InstanceNormalization()(conv9)
    conv10 = Conv2D(out_channels, 1)(conv9)

    if full_model:
        model = Model(inputs, conv10)
        return model
    return conv10, conv5
