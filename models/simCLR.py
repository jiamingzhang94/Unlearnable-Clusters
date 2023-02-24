import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18

class SimCLR(nn.Module):
    config = {'RN50': resnet50, 'RN18': resnet18}

    def __init__(self, feature_dim=128, num_classes=1000, name='RN50', temperature=0.07):
        super(SimCLR, self).__init__()
        self.name = name
        self.temperature = temperature
        backbone = self.config[name]()

        self.f = []
        for name, module in backbone.named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
            else:
                mlp_dim = module.in_features
        # encoder
        self.f = nn.Sequential(*self.f)

        # projection head
        self.g = nn.Sequential(nn.Linear(mlp_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, feature_dim))

        # classify head
        self.fc = nn.Linear(mlp_dim, num_classes)

        self.classify = False

    def forward(self, x):
        if self.classify:
            return self.forward_classify(x)
        else:
            return self.forward_pretrain(x)

    def forward_classify(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return self.fc(feature)

    def forward_pretrain(self, x):
        feature = self.f(x)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)
        feature = F.normalize(feature, dim=-1)

        labels = torch.cat([torch.arange(x.shape[0] // 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(x.device)

        similarity_matrix = torch.matmul(feature, feature.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(x.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        logits = logits / self.temperature

        return logits

