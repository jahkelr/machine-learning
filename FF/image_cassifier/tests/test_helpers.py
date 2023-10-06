import numpy as np
from image_classifier import helpers
import torch


def test_overlay_y_on_x():
    x = torch.rand(1200).reshape(3, 20, 20)
    y = np.random.choice(range(10))
    helpers.imshow(x, "pic", x.shape)
    x_ = helpers.overlay_y_on_x(x, y)
    helpers.imshow(x_, "pic", x.shape)
    # Label pixel is white
    assert x_[:, y, :1].mean().item() > 0.9


def test_format():
    x = torch.rand(3072)
    shape = (3, 32, 32)
    helpers.imshow(x, "pic", shape)
