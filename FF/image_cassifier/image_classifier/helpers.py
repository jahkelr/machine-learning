import matplotlib.pyplot as plt
from random import randint


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]"""
    if len(x.shape) < 3:
        # Handle 2D case
        x_ = x.clone()
        x_[:, :10] *= 0.0

        x_[range(x.shape[0]), y] = x.max()
        return x_

    # blacken all dimensions
    x_ = x.clone()
    x_[:, :10, :1] *= 0.0
    x_[range(x.shape[0]), y, :1] = x.max()
    return x_


def format(img, shape):
    if len(shape) == 3:  # 3 color channels
        img = img / 2 + 0.5  # unnormalize
        img = img.numpy().reshape(shape)
    else:  # monochrome
        img = img / 2 + 0.5  # unnormalize
        img = img.numpy().reshape(shape)
    return img


def imshow(img, name, shape):
    img = format(img, shape)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(img.T)
    plt.show()


def visualize_sample(data, shape, name="", idx=0):
    if not idx:
        idx = randint(0, len(data) - 1)
    reshaped = data[idx].cpu()
    imshow(reshaped, name, shape)
