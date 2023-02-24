import torch
import torch.nn.functional as F
from enum import Enum
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel


class ImageAttack_PGD():
    # PGD
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def input_diversity(self, image):
        return image

    def attack(self, image, num_iters, targeted=False):
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)

        if targeted:
            scaler = -1
        else:
            scaler = 1

        epsilon_per_iter = self.epsilon / num_iters * 1.25

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            grad = self.get_grad()
            grad = self.normalize(grad)
            self.delta = self.delta.data + epsilon_per_iter * grad * scaler

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def run(self, net, images, anchors, num_iters, targeted=False):
        features = {}
        def hook(layer, inp, out):
            features['feat'] = inp[0]
        net.fc.register_forward_hook(hook)

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(images, num_iters, targeted)

        for i in range(num_iters):
            image_adv = next(attacker)
            net(image_adv)

            loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv



class ImageAttack_DI(ImageAttack_PGD):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        super(ImageAttack_DI, self).__init__(epsilon, norm_type, random_init, *args, **kwargs)
        self.resize_rate = kwargs.get('resize_rate', 1.10)
        self.diversity_prob = kwargs.get('diversity_prob', 0.3)

    def input_diversity(self, x):
        assert self.resize_rate >= 1.0
        assert self.diversity_prob >= 0.0 and self.diversity_prob <= 1.0

        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        # print(img_size, img_resize, resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[img_size, img_size])
        ret = padded if torch.rand(1) < self.diversity_prob else x
        return ret


class ImageAttack_MI(ImageAttack_PGD):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        super(ImageAttack_MI, self).__init__(epsilon, norm_type, random_init,  *args, **kwargs)
        self.momentum = kwargs.get('momentum', 0.9)

    def get_grad(self):
        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(self.delta)

        grad = self.delta.grad.clone()
        self.grad = self.grad * self.momentum + grad
        return self.grad


class ImageAttack_DIM(ImageAttack_DI, ImageAttack_MI):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, *args, **kwargs):
        super(ImageAttack_DIM, self).__init__(epsilon, norm_type, random_init, *args, **kwargs)

    def input_diversity(self, x):
        return ImageAttack_DI.input_diversity(self, x)

    def get_grad(self):
        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(self.delta)

        grad = self.delta.grad.clone()
        self.grad = self.grad * self.momentum + grad
        return self.grad


def get_attacker(method, epsilon, *args, **kwargs):
    if method == 'PGD':
        return ImageAttack_PGD(epsilon, *args, **kwargs)
    elif method == 'DIM':
        return ImageAttack_DIM(epsilon, *args, **kwargs)

    return None
