# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from model import Hourglass

def pixel_thanos(img, p=0.5):
    assert p > 0 and p < 1, 'The probability value should lie in (0, 1)'
    mask = torch.rand(512, 512)
    img[:, :, mask<p] = 0
    mask = mask > p
    mask = mask.repeat(1,3,1,1)
    return img, mask

lr = 1e-2
device = 'cuda'

hg_net = Hourglass()
hg_net.to(device)
mse = nn.MSELoss()
optimizer = optim.Adam(hg_net.parameters(), lr=lr)

n_iter = 4000
images = []
losses = []
to_tensor = tv.transforms.ToTensor()
z = torch.Tensor(np.mgrid[:512, :512]).unsqueeze(0).to(device) / 512

x = Image.open('lena.png')
x = to_tensor(x).unsqueeze(0)
x, mask = pixel_thanos(x, 0.8)
mask = mask.to(device)
x = x.to(device)

for i in range(n_iter):
    optimizer.zero_grad()
    y = hg_net(z)
    loss = mse(x, y*mask)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    if (i+1)%25 == 0 or i==0:
        with torch.no_grad():
            out = x + y * ~mask
            out = out[0].cpu().detach().permute(1,2,0)*255
            out = np.array(out, np.uint8)
            images.append(out)
    if (i+1)%50 == 0:
        print('Iteration: {} Loss: {:.07f}'.format(i+1, losses[-1]))

plt.imsave('final.jpg', out)
plt.imsave('start.jpg', x[0].cpu().detach().permute(1,2,0).numpy())
plt.plot(losses)