from torch.utils.data import Dataset
import os
from PIL import Image
import torch

def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DataFolderWithLabel(Dataset):
    def __init__(self, root, pred_idx, transform=None):
        self.labels = []
        self.images = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name)
            for file_name in sorted(os.listdir(os.path.join(root, class_name))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, class_name, file_name))
                self.labels.append(label)

        if pred_idx is None:
            self.pred_idx = self.labels
        else:
            self.pred_idx = pred_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label, self.pred_idx[idx]


class DataFolderWithClassNoise(Dataset):
    def __init__(self, root, pred_idx, transform=None, noise=None):
        self.labels = []
        self.images = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name)
            for file_name in sorted(os.listdir(os.path.join(root, class_name))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, class_name, file_name))
                self.labels.append(label)

        if noise is None:
            self.noise = torch.zeros((1, 3, 112, 112))
            self.noise.uniform_(0, 1)
            self.noise = self.noise.repeat(self.num_classes, 1, 1, 1)
        else:
            self.noise = noise

        self.pred_idx = pred_idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        image = torch.clamp(image + self.noise[self.pred_idx[idx]], 0, 1)
        return image, label, self.pred_idx[idx]


class DataFolderWithOneClass(Dataset):
    def __init__(self, root, cluster_idx, pred_idx, transform=None, offset=0):
        self.images = []
        self.label = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name) - offset
            for file_name in sorted(os.listdir(os.path.join(root, class_name))):
                if not is_image_file(file_name):
                    continue
                self.images.append(os.path.join(root, class_name, file_name))
                self.label.append(label)

        self.pred_idx = pred_idx
        self.cluster_idx = cluster_idx

        self.images = [self.images[i] for i, eq in enumerate(pred_idx == cluster_idx) if eq]
        self.label = [self.label[i] for i, eq in enumerate(pred_idx == cluster_idx) if eq]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.label[idx], self.cluster_idx
