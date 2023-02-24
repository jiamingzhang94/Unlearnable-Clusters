import torch
import random

import numpy as np
from torchvision import transforms

from .gaussianSmoothing import GaussianSmoothing

class MixUp(torch.nn.Module):
    def __init__(self, lam=0.9):
        super(MixUp, self).__init__()
        self.lam = lam

    def forward(self, x):
        return self.lam * x + (1 - self.lam) * x[torch.randperm(x.shape[0])]


class CutMix(torch.nn.Module):
    def __init__(self, lam=0.9):
        super(CutMix, self).__init__()
        self.lam = lam

    def forward(self, x):
        mask = torch.zeros_like(x)
        height = int(x.shape[-2] * (1-self.lam)**0.5)
        width = int(x.shape[-1] * (1-self.lam)**0.5)

        top = random.randint(0, x.shape[-2] - height)
        left = random.randint(0, x.shape[-1] - width)

        mask[..., top: top+height, left: left+width] = 1.0

        return (1-mask) * x + mask * x[torch.randperm(x.shape[0])]


class CutOut(torch.nn.Module):
    def __init__(self, lam=0.9):
        super(CutOut, self).__init__()
        self.lam = lam

    def forward(self, x):
        mask = torch.zeros_like(x)
        height = int(x.shape[-2] * (1 - self.lam) ** 0.5)
        width = int(x.shape[-1] * (1 - self.lam) ** 0.5)

        top = random.randint(0, x.shape[-2] - height)
        left = random.randint(0, x.shape[-1] - width)

        mask[..., top: top + height, left: left + width] = 1.0

        return (1 - mask) * x


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
