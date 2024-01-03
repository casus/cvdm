import numpy as np
from einops import rearrange


def get_virus_signal_batch(XTrain, YTrain, p_size, b_size, center=None):
    r_idx = np.random.randint(0, XTrain.shape[0], b_size)
    if center is None:
        center_x = np.random.randint(p_size // 2, XTrain.shape[2] - p_size // 2)
        center_y = np.random.randint(p_size // 2, XTrain.shape[1] - p_size // 2)
    else:
        center_x = center[0]
        center_y = center[1]
    fx = np.expand_dims(
        XTrain[r_idx, center_y - p_size // 2:center_y + p_size // 2,
        center_x - p_size // 2:center_x + p_size // 2], -1)
    fy = np.expand_dims(
        YTrain[r_idx, center_y - p_size // 2:center_y + p_size // 2,
        center_x - p_size // 2:center_x + p_size // 2], -1)
    return fx, fy


def sample_norm_01(x):
    x = np.float32(x)
    n_x = (x - np.amin(x, axis=(1, 2), keepdims=True)) / (
            np.amax(x, axis=(1, 2), keepdims=True) - np.amin(x, axis=(1, 2), keepdims=True))

    return n_x * 2 - 1


def center_crop(x, crop_size=2048):
    x_center = x.shape[2] // 2
    y_center = x.shape[1] // 2

    return x[:, y_center - crop_size // 2:y_center + crop_size // 2,
           x_center - crop_size // 2:x_center + crop_size // 2]


def get_dataset(dataset_type='hsv'):
    if dataset_type == 'hsv':
        XTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/hsv/negative_control/x.npz")['arr_0'],
                             crop_size=2000)
        YTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/hsv/negative_control/y.npz")['arr_0'],
                             crop_size=2000)
    elif dataset_type == 'iav':
        XTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/iav/negative_control/x.npz")['arr_0'],
                             crop_size=2000)
        YTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/iav/negative_control/y.npz")['arr_0'],
                             crop_size=2000)

    elif dataset_type == 'rhv':
        XTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/rhv/negative_control/x.npz")['arr_0'],
                             crop_size=2000)
        YTrain = center_crop(np.load("/media/gabriel/data_hdd/vsign/rhv/negative_control/y.npz")['arr_0'],
                             crop_size=2000)
    elif dataset_type == 'hadv':
        data_hadv = np.load("/hdd1/full_dataset_lysis.npz")['x']
        XTrain = rearrange(data_hadv[..., 0, 45:49], 's h w t -> (s t) h w')
        YTrain = rearrange(data_hadv[..., 1, 45:49], 's h w t -> (s t) h w')


    else:
        print('No such dataset')

    XTrain = sample_norm_01(XTrain)
    YTrain = sample_norm_01(YTrain)

    return XTrain, YTrain
