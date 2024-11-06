import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def cord2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    return sum(cord2d(x, k) for x, k in zip(X, K))

def corr2d_multi_out(X, K):
    return torch.stack([corr2d_multi_in(X ,k) for k in K], 0)

image_path = '/Users/hendrick/Desktop/WechatIMG215.jpg'
image = Image.open(image_path)

preprocess = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = preprocess(image)

k_1 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1 ,-1]])
k_2 = torch.stack((k_1, k_1, k_1), 0)
K = torch.stack((k_2, k_2, k_2), 0)

result = corr2d_multi_out(image_tensor, K)
image_np = result.permute(1, 2, 0).numpy().clip(0, 1)
plt.imshow(image_np)
plt.axis('off')
plt.show()