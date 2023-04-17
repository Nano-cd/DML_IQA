from torchvision.models import densenet121
import torch.nn as nn
import torch


def _weights_init_kaiming(m):
    # classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


def _weights_init_xavier(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class Dense_net121(nn.Module):
    def __init__(self, num_classes, init_mode):
        super(Dense_net121, self).__init__()
        base_model = densenet121(pretrained=True)
        self.base = nn.Sequential(*list(base_model.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
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
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out, features
