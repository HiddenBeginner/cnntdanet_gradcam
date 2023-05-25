import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from astropy.io import fits


def seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_data(dir_data):
    trans_path = glob(f'{dir_data}/transient/*')
    trans_label = [1]*len(trans_path)
    trans_id = [f'transient_#{i}' for i in range(len(trans_path))]

    bogus_path = glob(f'{dir_data}/bogus/*')
    bogus_label= [0]*len(bogus_path)
    bogus_id = [f'bogus_#{i}' for i in range(len(bogus_path))]

    data_id = trans_id + bogus_id
    data_path = trans_path+bogus_path
    data_label = trans_label+bogus_label

    data_dict = {
            'ID' : data_id,
            'Label' : data_label,
            'Path' : data_path
            }

    return pd.DataFrame(data_dict)


def min_max_normalization(x):
    minimum = np.min(x)
    maximum = np.max(x)
    x_normalized = (x - minimum) / (maximum - minimum)

    return x_normalized


def load_fits(fpath):
    img = fits.getdata(fpath)         # np.ndarray of shape (38, 38)
    img = img[..., np.newaxis]        # (height, width, n_channels)
    img = min_max_normalization(img)  # MinMaxNormalization

    return img


def get_img(dataframe):
    img_list = []
    for path in dataframe['Path']:
        img_list.append(load_fits(path))
    np_img = np.array(img_list)
    return np_img
