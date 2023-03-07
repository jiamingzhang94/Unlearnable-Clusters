import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
from pathlib import Path
import ruamel.yaml as yaml
import argparse
from logger import MetricLogger
from utils import get_surrogate, get_target, normalize_list
from dataset.dataCluster import DataFolderWithLabel, DataFolderWithClassNoise
from models.generator import ResnetGenerator
from tqdm import tqdm


def train_gnet(args, config):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[config['normalize']]

    net = get_surrogate(config['model'], config['num_classes']).eval().to(args.device)
    # sd = torch.load(config['checkpoint'], map_location='cpu')
    # net.load_state_dict(sd)

    cluster = torch.load(config['cluster'], map_location='cpu')
    num_clusters = cluster['centers'].shape[0]

    train_dataset = DataFolderWithLabel(config['dataset']['config']['train'], cluster['pred_idx'], train_transform)
    train_loader = DataLoader(train_dataset, batch_size=256, num_workers=8)

    for cluster_idx in range(num_clusters):
        noise = torch.zeros((1, 3, 224, 224))
        noise.uniform_(0, 1)
        noise = noise.to(args.device)

        g_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu')
        g_net.to(args.device)

        optimizer = torch.optim.Adam(g_net.parameters(), lr=config['lr'], weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        logger = MetricLogger()

        features = {}
        def hook(layer, inp, out):
            features['feat'] = inp[0]
        net.fc.register_forward_hook(hook)

        for epoch in range(config['num_epoch']):
            g_net.train()
            header = 'Class idx {}\tTrain Epoch {}:'.format(cluster_idx, epoch)

            for images, _, _ in logger.log_every(train_loader, 50, header=header):
                images = images.to(args.device)
                delta_im = g_net(noise).repeat(images.shape[0], 1, 1, 1)

                if config['norm'] == 'l2':
                    temp = torch.norm(delta_im.view(delta_im.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * config['epsilon'] / temp
                else:
                    delta_im = torch.clamp(delta_im, -config['epsilon'] / 255., config['epsilon'] / 255)

                images_adv = torch.clamp(images + delta_im, 0, 1)
                target_labels = (torch.ones(len(images)).long() * cluster_idx + config['target_offset']) % num_clusters
                target_labels = target_labels.to(args.device)
                anchors = torch.stack([cluster['centers'][i] for i in target_labels], dim=0).to(args.device)

                net(normalize(images_adv))
                loss = criterion(features['feat'].log_softmax(dim=-1), anchors.softmax(dim=-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                logger.meters['train_loss'].update(loss.item(), n=len(images))

            with torch.no_grad():
                perturbation = g_net(noise)
            torch.save({'state_dict': g_net.state_dict(), 'init_noise': noise, 'perturbation': perturbation}, os.path.join(config['output_dir'], f'perturbation_{cluster_idx}.pth'))
            logger.clear()


def train(args, config):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    normalize = normalize_list[config['normalize']]

    num_classes = config['dataset']['config']['num_classes']

    train_dataset = DataFolderWithLabel(config['ae_dir'], None, transform=train_transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'], num_workers=8)

    test_dataset = DataFolderWithLabel(config['dataset']['config']['test'], None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=8)

    net = get_target(config['model'], num_classes).to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['num_epoch'] * len(train_loader), eta_min=1e-6)

    criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger()

    for epoch in range(config['num_epoch']):
        net.train()
        header = 'Train Epoch {}:'.format(epoch)

        for images, labels, _ in logger.log_every(train_loader, 50, header=header):
            images, labels = images.to(args.device), labels.to(args.device)

            logits = net(normalize(images))
            loss = criterion(logits, labels)

            pred_idx = torch.argmax(logits.detach(), 1)
            correct = (pred_idx == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            logger.meters['train_loss'].update(loss.item(), n=len(images))
            logger.meters['train_acc'].update(correct / len(images), n=len(images))

    net.eval()
    header = 'Test Epoch {}:'.format(epoch)
    for images, labels, _ in logger.log_every(test_loader, 50, header=header):
        images, labels = images.to(args.device), labels.to(args.device)

        with torch.no_grad():
            logits = net(normalize(images))
            loss = criterion(logits, labels)

        pred_idx = torch.argmax(logits.detach(), 1)
        correct = (pred_idx == labels).sum().item()

        logger.meters['test_loss'].update(loss.item(), n=len(images))
        logger.meters['test_acc'].update(correct / len(images), n=len(images))

    torch.save({'state_dict': net.state_dict()}, os.path.join(config['output_dir'], 'checkpoint.pth'))
    logger.clear()


def generate(args, config):
    normalize = normalize_list[config['normalize']]
    num_classes = config['dataset']['config']['num_classes']

    cluster = torch.load(config['cluster'], map_location='cpu')
    num_clusters = cluster['centers'].shape[0]

    noise = []
    for i in range(num_clusters):
        noise.append(torch.load(os.path.join(config['perturbation_dir'], f'perturbation_{i}.pth'), map_location='cpu')['perturbation'])
    noise = torch.cat(noise, dim=0)
    noise = torch.clamp(noise, -config['epsilon'] / 255., config['epsilon'] / 255)
    train_dataset = DataFolderWithClassNoise(config['dataset']['config']['train'], cluster['pred_idx'], noise=noise, resize_type=config['resize_type'])
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8)

    count = [0 for _ in range(config['dataset']['config']['num_classes'])]
    output_dir = config['ae_dir']
    print(output_dir)
    for i in range(len(count)):
        Path(os.path.join(output_dir, str(i))).mkdir(parents=True, exist_ok=True)
    print('Done floder')

    logger = MetricLogger()
    header = 'Generate cluster-wise UEs:'

    count = [0 for _ in range(num_classes)]
    for i in range(len(count)):
        Path(os.path.join(config['output_dir'], '..', 'ae', str(i))).mkdir(parents=True, exist_ok=True)

    for images, ground_truth, _ in train_loader:
        images_adv = images

        ground_truth = ground_truth.tolist()

        for i in range(len(images)):
            gt = ground_truth[i]
            save_image(images_adv[i], os.path.join(config['output_dir'], '..', 'ae', str(gt), f'{count[gt]}.png'))
            count[gt] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/stage_2.yaml')
    parser.add_argument('--experiment', '-e', type=str, default='uc_pets_cliprn50_rn18')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage', type=int, default=2)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)[args.experiment]
    data_config = yaml.load(open(config['data_config'], 'r'), Loader=yaml.Loader)[config['dataset']]
    config['dataset'] = {'name': config['dataset'], 'config': data_config}
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

    if args.stage == 1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], '..', 'config.yaml'), 'w+'))
        train_gnet(args, config)
    elif args.stage == 2:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        yaml.dump(config, open(os.path.join(config['output_dir'], 'config.yaml'), 'w+'))
        generate(args, config)
        train(args, config)
    else:
        raise KeyError
