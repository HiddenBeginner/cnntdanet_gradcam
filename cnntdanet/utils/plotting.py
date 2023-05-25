import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def visualize_image_samples(samples, targets=None, ncols=4, scale=3):
    """
    samples (np.ndarray) : np.ndarray with the shape [batch_size, height, width, channels]
    targets (np.ndarray or list, optional) : Sequence of labels corresponding to samples. Default=None.
    ncols (int, optional) : Number of images displayed in each row of the grid. 
    scale (float, optional) : Value for the size of each sample 
    """
    nrows = int(np.ceil(len(samples) / ncols))

    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(scale * ncols, scale * nrows))
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            if count >= len(samples):
                axes[i][j].axis('off')
                continue

            if samples.shape[-1] == 1:
                axes[i][j].imshow(samples[count], cmap='gray')
            else:
                axes[i][j].imshow(samples[count])
            axes[i][j].axis('off')
            if targets is not None:
                axes[i][j].set_title(f'Target: {targets[count]}')

            count += 1
    fig.tight_layout()


def plot_learning_curve(history, dir_save=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    e = np.arange(len(history['loss']), dtype=np.int64)

    fig.suptitle("Learning curves", fontsize=18)
    axes[0].plot(e, history['loss'], 'bo-', label='train')
    axes[0].plot(e, history['val_loss'], 'ro-', label='valid')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(e, history['acc'], 'bo-', label='train')
    axes[1].plot(e, history['val_acc'], 'ro-', label='valid')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid()

    if dir_save is not None:
        plt.savefig(f'{dir_save}/learning-curve.pdf', dpi=250)
