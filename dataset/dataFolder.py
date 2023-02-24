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
    def __init__(self, root, transform=None):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


class DataFolderWithClassNoise(Dataset):
    def __init__(self, root, transform=None, name_mapping=None, offset=0, noise=None):
        self.labels = []
        self.images = []
        self.transform = transform

        for class_name in sorted(os.listdir(root)):
            label = int(class_name) - offset
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label, self.noise[label]

class DataFolderWithOneClass(Dataset):
    def __init__(self, root, label, transform=None):
        self.label = label
        self.images = []
        self.transform = transform

        class_name = str(label)

        for file_name in sorted(os.path.join(root, class_name)):
            if not is_image_file(file_name):
                continue
            self.images.append(os.path.join(root, class_name, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, self.label


class DataFolderWithContrastiveGenerator(DataFolderWithLabel):
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = 0

        image = [self.transform(image) for _ in range(2)]
        label = [label for _ in range(2)]
        return image, label

