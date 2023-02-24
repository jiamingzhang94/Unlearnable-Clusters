import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import os
import shutil
from pathlib import Path
import ruamel.yaml as yaml

from logger import MetricLogger
from utils import get_model, normalize_list

from dataset.dataFolder import DataFolderWithLabel, DataFolderWithOneClass

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import json

def dataset_median_embedding(args, config):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[args.normalize]

    num_classes = len(os.listdir(config[args.dataset]['test']['path']))
    net = get_model(args.model, num_classes).eval().to(args.device)
    sd = torch.load(args.checkpoint, map_location='cpu')
    sd = sd.get('state_dict', sd)
    net.load_state_dict(sd)

    result = []

    for i in range(num_classes):
        train_dataset = DataFolderWithOneClass(config[args.dataset]['train']['path'], i, transform)
        train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8)

        features = []
        def hook(layer, inp, out):
            features.append(inp[0].cpu())
        net.fc.register_forward_hook(hook)

        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            with torch.no_grad():
                net(normalize(images))

        features = torch.cat(features, dim=0)
        dis = squareform(pdist(features.numpy())).sum(axis=1)
        dis = torch.tensor(dis)
        idx = torch.argmin(dis)
        result.append(features[idx])

    torch.save(result, os.path.join(args.output_dir, f'{args.dataset}_{args.model}_median_embed.pth'))


def data_cluster(args, config):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[config['normalize']]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], transform)
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)

    sd = torch.load(config['checkpoint'], map_location='cpu')
    sd = sd.get('state_dict', sd)
    net = get_model(config['model'], config['num_classes']).eval().to(args.device)
    net.load_state_dict(sd)

    features = []

    def hook(layer, inp, out):
        features.append(inp[0].cpu())

    net.fc.register_forward_hook(hook)

    for images, labels in train_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        with torch.no_grad():
            net(normalize(images))

    features = torch.cat(features, dim=0)
    classifier = KMeans(n_clusters=config['num_clusters'])
    pred_idx = classifier.fit_predict(features.numpy())

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], transform)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=8)

    features = []
    for images, labels in test_loader:
        images, labels = images.to(args.device), labels.to(args.device)
        with torch.no_grad():
            net(normalize(images))

    features = torch.cat(features, dim=0)
    pred_idx_test = classifier.predict(features.numpy())

    result = {'pred_idx': torch.tensor(pred_idx), 'pred_idx_test': pred_idx_test, 'centers': torch.tensor(classifier.cluster_centers_)}
    num_clusters = result['centers'].shape[0]
    torch.save(result, os.path.join(config['output_dir'], f"{config['dataset']['name']}_{config['model'].lower()}_cluster{num_clusters}.pth"))


def datafolder_rename(args, config):
    train_dir = config['dataset']['config']['train']
    test_dir = config['dataset']['config']['test']

    class_list = sorted(os.listdir(train_dir))
    new_class = list(map(str, range(len(class_list))))

    name_mapping = {}
    name_mapping['map'] = {k: v for k, v in zip(class_list, new_class)}
    name_mapping['inv'] = {k: v for k, v in zip(new_class, class_list)}

    json.dump(name_mapping, open(os.path.join(train_dir, '..', 'name_mapping.json'), 'w+'))

    os.rename(train_dir, train_dir+'_raw')
    os.rename(test_dir, test_dir+'_raw')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    for i in range(len(class_list)):
        shutil.copytree(os.path.join(train_dir+'_raw', class_list[i]), os.path.join(train_dir, new_class[i]))
        shutil.copytree(os.path.join(test_dir+'_raw', class_list[i]), os.path.join(test_dir, new_class[i]))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=str)

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--function', '-f', type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.function]
    data_config = yaml.load(open(config['data_config'], 'r'), Loader=yaml.Loader)[config['dataset']]
    config['dataset'] = {'name': config['dataset'], 'config': data_config}

    if args.function.startswith('median'):
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        dataset_median_embedding(args, config)
    elif args.function.startswith('cluster'):
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        data_cluster(args, config)
    elif args.function.startswith('rename'):
        datafolder_rename(args, config)
