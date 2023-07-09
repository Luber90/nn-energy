import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from unet import *


img2 = Image.open('unlabeled2017/000000002437.jpg').convert("RGB")
img = Image.open('unlabeled2017/000000002272.jpg').convert('RGB')
plt.imshow(img)

SIZE = 128

img = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)(img)

img_lab = rgb2lab(img).astype("float32")
img_lab = transforms.ToTensor()(img_lab)
L = img_lab[[0], ...] / 50. - 1.
ab = img_lab[[1, 2], ...] / 128.
L = L[None,:]
ab = ab[None,:]
true_L = (L + 1)*50.
true_ab = ab*128.
true_pic = torch.cat([true_L, true_ab], dim=1)
true_pic = true_pic.permute(0, 2, 3, 1)
#true_pic = true_pic.resize(1,256, 256, 3)
# reconctructed = true_pic.numpy()
reconstructed = lab2rgb(true_pic)

device = torch.device("cuda:0")

model = UNet()
model.load_state_dict(torch.load("model_output/model"))
model.to(device)
L = L.to(device)
model.eval()
pred_ab = model(L)
pred_ab = pred_ab.cpu()
pred_ab = pred_ab*128
pred_pic = torch.cat([true_L, pred_ab], dim=1)
pred_pic = pred_pic.permute(0, 2, 3, 1)
pred_pic = pred_pic.detach().numpy()
pred_rec = lab2rgb(pred_pic)

plt.subplot(2, 2, 1)
plt.imshow(pred_rec[0])
plt.subplot(2, 2, 2)
plt.imshow(reconstructed[0])
plt.subplot(2, 2, 3)
true_L = true_L.permute(0, 2, 3, 1)
plt.imshow(-1*true_L[0], cmap='Greys')


plt.show()