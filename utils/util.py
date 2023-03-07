from models import *
from dataset import *
import torchvision
import torch
from torchvision import transforms

normalize_list = {'clip': transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                  'general': transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                  'imagenet': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))  # args.lr = 0.1 ,
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_surrogate(name, num_classes=1000):
    if name == 'RN50':
        model = resnet50(pretrained=True)
        torch.save(model.state_dict(), 'checkpoints/RN50_imagenet.pth')
    elif name == 'CLIPRN50':
        model = ClipResnet(name='RN50', num_classes=num_classes)
        model.load_pretrain()
        torch.save(model.state_dict(), 'checkpoints/RN50_clip.pth')
    else:
        raise f'Model {name} Not Found'
    return model


def get_target(name, num_classes=1000):
    if name == 'RN18':
        model = resnet18(num_classes=num_classes)
    elif name == 'SimCLRRN50':
        model = SimCLR(name='RN50', num_classes=num_classes)
    elif name == 'regnet':
        model = torchvision.models.regnet_x_1_6gf(num_classes=num_classes)
    elif name == 'efficientnet_b1':
        model = torchvision.models.efficientnet_b1(num_classes=num_classes)
    else:
        raise (f'Model {name} Not Found')
    return model
