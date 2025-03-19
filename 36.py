import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize =(num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i][j]
            ax.imshow(imgs[i * num_cols + j], cmap='gray')
            ax.axis('off')
            if titles:
                ax.set_title(titles[i * num_cols + j])
    
    plt.show()

image = d2l.Image.open('../WechatIMG215.jpg')

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)

apply(image, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))