from model_hub import inceptionresnetv2
import torch.nn as nn
import torch


def _weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


def _weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class Inceptionresnet_v2(nn.Module):
    def __init__(self, num_classes, init_mode):
        super(Inceptionresnet_v2, self).__init__()
        base_model = inceptionresnetv2.inceptionresnetv2(num_classes=1, pretrained='imagenet')
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_out = nn.Dropout()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1536, 512),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes),
        )

        if init_mode == 'kaiming_norm':
            for m in self.fc:
                m.apply(_weights_init_kaiming)
        else:
            for m in self.fc:
                m.apply(_weights_init_xavier)

    def forward(self, x):
        features = self.base(x)
        out = self.avg_pool(features)
        out = self.drop_out(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out, features
