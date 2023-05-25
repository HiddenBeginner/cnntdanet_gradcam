from tensorflow import keras


def make_cnn_tda_net(img_network, tda_network, head, input_shape):
    """
    Build a custom CNN-TDA Net from given img_network, tda_network, and head.

    Args:
        img_network (keras.Model): _description_
        tda_network (keras.Model): _description_
        head (keras.Model): _description_
        input_shape (dict): a dictionary that has two keys 'img' and 'tda' and 
        the corresponding values are the shapes of images and tda features

    Returns:
        keras.Model: A custom CNN-TDA Net
    """
    input_img = keras.layers.Input(input_shape['img'])
    input_tda = keras.layers.Input(input_shape['tda'])

    # Define the forward pass. Actual model is defined by run.
    feature_img = img_network(input_img)
    feature_tda = tda_network(input_tda)
    concat = keras.layers.concatenate([feature_img, feature_tda])
    out = head(concat)

    return keras.Model(inputs=[input_img, input_tda], outputs=[out])


def get_cnn_tda_net(method, input_shape, n_classes):
    img_network = get_2d_cnn(input_shape['img'], name='img_network')
    if method == 'persistence-image':
        tda_network = get_2d_cnn(input_shape['tda'], name='tda_network')
    elif method in ['betti-curve', 'persistence-landscape']:
        tda_network = get_1d_cnn(input_shape['tda'], name='tda_network')
    head = get_classification_head(n_classes)

    input_img = keras.layers.Input(input_shape['img'])
    input_tda = keras.layers.Input(input_shape['tda'])

    # Define the forward pass. Actual model is defined by run.
    feature_img = img_network(input_img)
    feature_tda = tda_network(input_tda)
    concat = keras.layers.concatenate([feature_img, feature_tda])
    out = head(concat)

    return keras.Model(inputs=[input_img, input_tda], outputs=[out])


def get_cnn_net(input_shape, n_classes):
    img_network = get_2d_cnn(input_shape, name='img_network')
    head = get_classification_head(n_classes)

    return keras.models.Sequential([img_network, head])


def get_wide_cnn_net(input_shape, n_classes):
    img_network = get_wide_2d_cnn(input_shape, name='img_network')
    head = get_classification_head(n_classes)

    return keras.models.Sequential([img_network, head])


def get_1d_cnn(input_shape, name='tda_network'):
    cnn = keras.models.Sequential(name=name)
    cnn.add(keras.layers.InputLayer(input_shape=input_shape))
    cnn.add(keras.layers.BatchNormalization())
    for rate in (1, 2, 4, 8) * 2:
        cnn.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=rate))
        cnn.add(keras.layers.Dropout(0.1))
    cnn.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    cnn.add(keras.layers.Flatten())

    return cnn


def get_2d_cnn(input_shape, name='img_network'):
    cnn = keras.models.Sequential([
        keras.layers.Conv2D(16, 3, activation="relu", padding='same', input_shape=input_shape),
        keras.layers.Conv2D(32, 3, activation="relu", padding='same'),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten()
    ], name=name)

    return cnn


def get_wide_2d_cnn(input_shape, name='img_network'):
    cnn = keras.models.Sequential([
        keras.layers.Conv2D(32, 3, activation="relu", padding='same', input_shape=input_shape),
        keras.layers.Conv2D(64, 3, activation="relu", padding='same'),
        keras.layers.AveragePooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten()
    ], name=name)

    return cnn


def get_classification_head(n_classes, name='head'):
    head = keras.models.Sequential([
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ], name=name)

    return head
