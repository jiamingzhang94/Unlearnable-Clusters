import torch
from .clip.model import ModifiedResNet
from .clip.clip import load


class ClipResnet(torch.nn.Module):
    config = {'RN50':
                  {'vision_layers': (3, 4, 6, 3), 'embed_dim': 1024,
                   'vision_heads': 32, 'image_resolution': 224, 'vision_width': 64},
              'RN101':
                  {'vision_layers': (3, 4, 23, 3), 'embed_dim': 512,
                   'vision_heads': 32, 'image_resolution': 224, 'vision_width': 64}}

    def __init__(self, name='RN50', num_classes=1000):
        super(ClipResnet, self).__init__()

        assert name in self.config.keys()
        self.name = name

        self.visual_encoder = ModifiedResNet(
            layers=self.config[name]['vision_layers'],
            output_dim=self.config[name]['embed_dim'],
            heads=self.config[name]['vision_heads'],
            input_resolution=self.config[name]['image_resolution'],
            width=self.config[name]['vision_width'],
        )

        self.fc = torch.nn.Linear(self.config[name]['embed_dim'], num_classes)

    def load_pretrain(self):
        temp, _ = load(self.name, 'cpu')
        pretrained = temp.visual.state_dict()

        visual_encoder_dict = self.visual_encoder.state_dict()
        state_dict = {k: v for k, v in pretrained.items() if k in visual_encoder_dict.keys()}
        visual_encoder_dict.update(state_dict)
        self.visual_encoder.load_state_dict(visual_encoder_dict)

    def forward(self, image):
        x = self.visual_encoder(image)
        x = self.fc(x)
        return x

