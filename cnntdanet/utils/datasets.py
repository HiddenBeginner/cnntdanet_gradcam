import numpy as np
import pandas as pd
import tensorflow as tf

from ..tda import get_tda_pipeline


def prepare_dataset(dataset, dir_data=None, method=None, **kwargs):
    if dataset == 'fashion-mnist':
        dataset = _prepare_fashion_mnist(method, **kwargs)

    if dataset == 'skin-cancer':
        dataset = _prepare_skin_cancer(dir_data, method, **kwargs)

    return dataset


def _prepare_fashion_mnist(method=None, **kwargs):
    data = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = data.load_data()

    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    X_train = X_train.astype(np.float32) / 255.0
    X_train = X_train[..., np.newaxis]
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    X_test  = X_test.astype(np.float32) / 255.0
    X_test  = X_test[..., np.newaxis]
    y_test  = tf.keras.utils.to_categorical(y_test, 10)

    dataset = {'X_img': X_train, 'y': y_train, 'X_img_test': X_test, 'y_test': y_test}
    if method is not None:
        pipeline = get_tda_pipeline(method, **kwargs)
        X_tda = pipeline.fit_transform(X_train)
        X_tda_test  = pipeline.fit_transform(X_test)

        dataset['X_tda'] = X_tda
        dataset['X_tda_test']  = X_tda_test

    return dataset


def _prepare_skin_cancer(dir_data, method=None, **kwargs):
    df = pd.read_csv(dir_data)

    X = df.drop('label', axis=1).values.reshape((-1, 28, 28, 1))
    X = X.astype(np.float32) / 255.0

    y = df['label'].values.astype(np.int64)
    y = tf.keras.utils.to_categorical(y, 7)

    dataset = {'X_img': X, 'y': y}
    if method is not None:
        pipeline = get_tda_pipeline(method, **kwargs)
        X_tda = pipeline.fit_transform(X)

        dataset['X_tda'] = X_tda

    return dataset
