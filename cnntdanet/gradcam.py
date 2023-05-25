from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class GradCAMBase:
    def __init__(self, model):
        self.model = model
        self.__dict__['_cache'] = defaultdict(lambda: None)

    def to_heatmap(self):
        raise NotImplementedError

    def _get_grad_model(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError


class GradCAMOnCNN(GradCAMBase):
    def __init__(self, model, layer_name):
        super().__init__(model=model)
        self._grad_model = self._get_grad_model(self.model, layer_name)

    def to_heatmap(self, img, target_label=None, true_label=None):
        if img.ndim == 3:
            img = img[np.newaxis]

        self._cache['img'] = np.uint(255 * img)[0]
        self._cache['true_label'] = true_label
        self._cache['target_label'] = target_label
        self._cache['grad_maps'] = []
        self._cache['feature_maps'] = []

        local_heatmap = self._compute_local_heatmap(img, target_label)
        self._cache['heatmap'] = [local_heatmap]

    def visualize(self, alpha=0.4, label_decoder=None):
        if label_decoder is None:
            label_decoder = {
                self._cache['true_label']: self._cache['true_label'],
                self._cache['pred_label']: self._cache['pred_label'],
            }
        img = self._cache['img']
        heatmap = self._cache['heatmap'][0]
        cmap = 'gray' if img.shape[-1] == 1 else None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img, cmap=cmap)       # Image
        axes[1].imshow(heatmap, cmap='jet')  # Heatmap
        axes[2].imshow(img, cmap=cmap)       # Image + heatmap
        im = axes[2].imshow(heatmap, alpha=alpha, cmap='jet')  # Image + heatmap
        cax = fig.add_axes([0.91, 0.16, 0.01, 0.69])
        fig.colorbar(im, cax)

        fig.suptitle(f"Label: {label_decoder[self._cache['true_label']]} " +
                     f"| Prediction: {label_decoder[self._cache['pred_label']]}")
        for ax in axes:
            ax.axis('off')

    def _get_grad_model(self, model, layer_name):
        # Get models
        img_network = model.get_layer('img_network')
        head = model.get_layer('head')

        # Define forward pass
        seq_input = img_network.inputs
        flatten = img_network.output
        output = head(flatten)

        last_feature_maps = img_network.get_layer(layer_name).output

        return keras.models.Model([seq_input], [last_feature_maps, output])

    def _compute_local_heatmap(self, img, target_label=None):
        with tf.GradientTape() as tape:
            last_feature_maps, preds = self._grad_model(img)
            pred_label = tf.argmax(preds[0])
            self._cache['pred_label'] = pred_label.numpy()
            if target_label is None:
                target_label = pred_label
            class_channel = preds[:, target_label]

        # Taking the gradient of outputs w.r.t the last feature maps
        grads = tape.gradient(class_channel, last_feature_maps)

        # For each channel, taking the average of the gradient over spatial domain
        # This will indicate the channel importance
        # (0, 1, 2) means taking average along the axis 0, 1, 2 (# data, width, height)
        pooled_grads = tf.reduce_mean(grads, (0, 1, 2))

        # Since the batch size = 1, removing the useless first axis
        last_feature_maps = tf.squeeze(last_feature_maps)
        self._cache['grad_maps'].append(tf.squeeze(grads).numpy())
        self._cache['feature_maps'].append(last_feature_maps.numpy())

        # The weighted sum of feature maps along channels with weight channel importance
        heatmap = last_feature_maps @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        # [0, 1] to [0, 255]
        heatmap = heatmap.numpy()[..., np.newaxis]
        heatmap = np.uint8(255 * heatmap)

        # Matching the size of heatmap with the original image using PIL.Image object
        heatmap = keras.preprocessing.image.array_to_img(heatmap)  # np.ndarray -> PIL.Image
        heatmap = heatmap.resize((self._cache['img'].shape[1], self._cache['img'].shape[0]))  # Resizing
        heatmap = keras.preprocessing.image.img_to_array(heatmap)  # PIL.Image -> np.ndarr

        return heatmap


class GradCAMOnCNNTDANet(GradCAMBase):
    def __init__(self, model, local_layer_name, global_layer_name):
        super().__init__(model=model)
        self._grad_model = self._get_grad_model(self.model, local_layer_name, global_layer_name)

    def to_heatmap(self, inputs, target_label=None, true_label=None):
        # Grab the inputs
        self._cache['img'] = np.uint8(255 * inputs[0]).squeeze(0)
        self._cache['tda'] = inputs[1].squeeze(0)
        self._cache['target_label'] = target_label
        self._cache['true_label'] = true_label

        self._cache['grad_maps'] = []
        self._cache['feature_maps'] = []
        local_heatmap = self._compute_local_heatmap(inputs, target_label)
        global_heatmap = self._compute_global_heatmap(inputs, target_label)
        self._cache['heatmap'] = [local_heatmap, global_heatmap]

    def _compute_local_heatmap(self, inputs, target_label=None):
        with tf.GradientTape() as tape:
            local_layer_output, global_layer_output, preds = self._grad_model(inputs)
            pred_label = tf.argmax(preds[0])
            if target_label is None:
                target_label = pred_label
            class_channel = preds[:, target_label]

        grads = tape.gradient(class_channel, local_layer_output)
        pooled_grads = tf.reduce_mean(grads, (0, 1, 2))
        local_layer_output = tf.squeeze(local_layer_output)
        self._cache['grad_maps'].append(tf.squeeze(grads).numpy())
        self._cache['feature_maps'].append(local_layer_output.numpy())

        heatmap = local_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        heatmap = np.uint8(255 * heatmap)[..., np.newaxis]
        heatmap = keras.preprocessing.image.array_to_img(heatmap)  # np.ndarray -> PIL.Image
        heatmap = heatmap.resize((self._cache['img'].shape[1], self._cache['img'].shape[0]))  # Resizing
        heatmap = keras.preprocessing.image.img_to_array(heatmap)  # PIL.Image -> np.ndarray

        return heatmap

    def _compute_global_heatmap(self, inputs, target_label=None):
        # For global pipeline
        with tf.GradientTape() as tape:
            local_layer_output, global_layer_output, preds = self._grad_model(inputs)
            pred_label = tf.argmax(preds[0])
            self._cache['pred_label'] = pred_label.numpy()
            if target_label is None:
                target_label = pred_label
            class_channel = preds[:, target_label]
        grads = tape.gradient(class_channel, global_layer_output)
        pooled_grads = tf.reduce_mean(grads, (0, 1))
        global_layer_output = tf.squeeze(global_layer_output)
        self._cache['grad_maps'].append(tf.squeeze(grads).numpy())
        self._cache['feature_maps'].append(global_layer_output.numpy())

        heatmap = global_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()[..., np.newaxis]
        heatmap = np.hstack([heatmap] * 3)

        return heatmap

    def _get_grad_model(self, model, local_layer_name, global_layer_name):
        # Get models
        img_network = model.get_layer('img_network')
        tda_network = model.get_layer('tda_network')
        head = model.get_layer('head')

        # Define forward pass
        local_input = img_network.inputs
        local_flatten = img_network.output

        global_input = tda_network.inputs
        global_flatten = tda_network.output

        concat = model.get_layer('concatenate')([local_flatten, global_flatten])
        output = head(concat)

        local_layer_output = img_network.get_layer(local_layer_name).output
        global_layer_output = tda_network.get_layer(global_layer_name).output

        return keras.models.Model([local_input, global_input], [local_layer_output, global_layer_output, output])

    def visualize(self, alpha=0.4, save=None, label_decoder=None):
        if label_decoder is None:
            label_decoder = {
                self._cache['true_label']: self._cache['true_label'],
                self._cache['pred_label']: self._cache['pred_label'],
            }

        fig = plt.figure(figsize=(15, 5))

        # CNN
        ax_img1 = fig.add_axes([0.02, 0.2, 0.2, 0.60])
        ax_img1.imshow(self._cache['img'], cmap='gray')
        ax_img1.axis('off')
        ax_img1.set_title('Image', y=1.03)

        ax_img2 = fig.add_axes([0.27, 0.2, 0.2, 0.60])
        ax_img2.imshow(self._cache['img'], cmap='gray')
        ax_img2.imshow(self._cache['heatmap'][0], cmap='jet', alpha=alpha)
        ax_img2.axis('off')
        ax_img2.set_title('Grad-CAM', y=1.03)

        # TDA
        ax = fig.add_axes([0.55, 0.1, 0.40, 0.8])
        ax.plot(self._cache['tda'][:, 0], 'b', label='dim0')
        ax.plot(self._cache['tda'][:, 1], 'r', label='dim1')
        ax.set_xticks(np.arange(101)[::10], [f'{0.1 * i:.1f}' for i in range(11)])
        ax.grid()
        ax.legend()

        ax_mat = fig.add_axes([0.568, -0.335, 0.364, 0.9])
        ax_mat.matshow(self._cache['heatmap'][1].T, cmap='jet')
        ax_mat.axis('off')

        fig.suptitle(
            f"Label: {label_decoder[self._cache['true_label']]} | " +
            f"Prediction: {label_decoder[self._cache['pred_label']]}",
            fontsize=20,
            y=0.98
        )

        if save is not None:
            plt.savefig(fig, dpi=200)
