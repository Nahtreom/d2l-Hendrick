import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_numofwork():
    return 4

d2l.use_svg_display()

trans = transforms.ToTensor()
minst_train = torchvision.datasets.FashionMNIST(
    root='data', transform=trans, train=True, download=False
)
minst_test = torchvision.datasets.FashionMNIST(
    root='data', transform=trans, train=False, download=False 
)

X,y = next(iter(data.DataLoader(minst_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
#d2l.plt.show()

batch_size = 256
train_iter = data.DataLoader(minst_train, batch_size, shuffle=True, num_workers=get_dataloader_numofwork())

timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop(): .2f} sec')
