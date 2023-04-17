import argparse
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from utils import compute_metric
from torch.utils.tensorboard import SummaryWriter as sum_writer
import model_hub
import dataset.create_dataset
from model_hub import Alexnet_modify, VGG16_modify, resnet34_modify, mobilenet_v2_modify, Inceptionresnetv2_modify, \
    densenet121_modify

import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=str, default='ML', help='Mutual learning')
    parser.add_argument("--data_mode", type=str, default='test', help='train test')
    parser.add_argument('--data', type=str, default='LIVE_C')
    parser.add_argument("--train", type=bool, default=False)

    parser.add_argument("--network", type=str, default='Inceptionresnetv2'
                        , help='resnet34, vgg16, '
                               'densenet121, mobilenet_v2, alexnet, Inceptionresnetv2')

    parser.add_argument("--train_description1", type=str, default='ML8KS',
                        help='train_description')
    parser.add_argument("--train_description2", type=str, default='ML8KT',
                        help='train_description')
    parser.add_argument("--lamda", type=float, default=1)

    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt1', default='Inceptionresnet_v2ML8KS-00039.pt',
                        type=str, help='name of the first checkpoint to load')
    parser.add_argument('--ckpt2', default='Inceptionresnet_v2ML8KT-00039.pt',
                        type=str, help='name of the second checkpoint to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--number_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=10)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--init1", type=str, default='kaiming_norm')
    parser.add_argument("--init2", type=str, default='xavier')

    return parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.global_step = 0
        self.config = config
        self.train_mode = config.train_mode
        self.data_mode = config.data_mode
        # initialize the data_loader
        if self.config.data == 'kon10k1000':
            self.train_dataloader = dataset.create_dataset.kon10k_1000(self.config)
        elif self.config.data == 'kon10k2000':
            self.train_dataloader = dataset.create_dataset.kon10k_2000(self.config)
        elif self.config.data == 'kon10k3000':
            self.train_dataloader = dataset.create_dataset.kon10k_3000(self.config)
        elif self.config.data == 'kon10k8000':
            self.train_dataloader = dataset.create_dataset.kon10k_8000(self.config)
        elif self.config.data == 'kadid10k1000':
            self.train_dataloader = dataset.create_dataset.kadid10k_1000(self.config)
        elif self.config.data == 'kadid10k2000':
            self.train_dataloader = dataset.create_dataset.kadid10k_2000(self.config)
        elif self.config.data == 'kadid10k3000':
            self.train_dataloader = dataset.create_dataset.kadid10k_3000(self.config)
        elif self.config.data == 'kadid10k8000':
            self.train_dataloader = dataset.create_dataset.kadid10k_8000(self.config)
        elif self.config.data == 'LIVE_C':
            self.train_dataloader = dataset.create_dataset.live_c(self.config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the model
        if config.network == 'resnet34':
            print('resnet34 model selected')
            self.model1 = resnet34_modify.Resnet34(num_classes=1, init_mode=self.config.init1)
            self.model2 = resnet34_modify.Resnet34(num_classes=1, init_mode=self.config.init2)
        elif config.network == 'vgg16':
            print('vgg16 model selected')
            self.model1 = VGG16_modify.VGG16(num_classes=1, init_mode=self.config.init1)
            self.model2 = VGG16_modify.VGG16(num_classes=1, init_mode=self.config.init2)
        elif config.network == 'Inceptionresnetv2':
            print('Inceptionresnetv2 model selected')
            self.model1 = Inceptionresnetv2_modify.Inceptionresnet_v2(num_classes=1, init_mode=self.config.init1)
            self.model2 = Inceptionresnetv2_modify.Inceptionresnet_v2(num_classes=1, init_mode=self.config.init2)
        elif config.network == 'densenet121':
            print('densenet121 model selected')
            self.model1 = densenet121_modify.Dense_net121(num_classes=1, init_mode=self.config.init1)
            self.model2 = densenet121_modify.Dense_net121(num_classes=1, init_mode=self.config.init2)
        elif config.network == 'alexnet':
            print('alexnet model selected')
            self.model1 = Alexnet_modify.Alex_net(num_classes=1, init_mode=self.config.init1)
            self.model2 = Alexnet_modify.Alex_net(num_classes=1, init_mode=self.config.init2)
        elif config.network == 'mobilenet_v2':
            print('mobilenet_v2 model selected')
            self.model1 = mobilenet_v2_modify.Mobilenet_v2(num_classes=1, init_mode=self.config.init1)
            self.model2 = mobilenet_v2_modify.Mobilenet_v2(num_classes=1, init_mode=self.config.init2)
        else:
            raise NotImplementedError("Not supported network, need to be added!")

        self.model1.to(self.device)
        self.model_name1 = type(self.model1).__name__ + self.config.train_description1
        self.model2.to(self.device)
        self.model_name2 = type(self.model2).__name__ + self.config.train_description2
        # print(self.model)
        # try load the model

        # initialize the loss function and optimizer
        self.start_epoch = 0
        self.lamda = config.lamda
        self.max_epoch = config.max_epoch
        self.loss_fn = torch.nn.MSELoss()
        self.ckpt_path = config.ckpt_path
        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=config.lr)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=config.lr)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1,
                                                          last_epoch=self.start_epoch - 1,
                                                          step_size=config.decay_interval,
                                                          gamma=config.decay_ratio)
        self.scheduler2 = torch.optim.lr_scheduler.StepLR(self.optimizer2,
                                                          last_epoch=self.start_epoch - 1,
                                                          step_size=config.decay_interval,
                                                          gamma=config.decay_ratio)
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name1 + str(self.config.split))
        self.logger = sum_writer(runs_path)
        if not config.train:
            ckpt1 = os.path.join(config.ckpt_path, config.ckpt1)
            ckpt2 = os.path.join(config.ckpt_path, config.ckpt2)
            self._load_checkpoint(ckpt1=ckpt1, ckpt2=ckpt2)

    def fit(self):
        if self.train_mode == 'ML':
            for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
                self._train_one_epoch(epoch)
                self.scheduler1.step()
                self.scheduler2.step()

    def _train_one_epoch(self, epoch):
        # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model1.train()
        self.model2.train()
        for _, (x, y) in enumerate(self.train_dataloader):
            x = Variable(x)
            y = Variable(y)

            x = x.to(self.device)
            y = y.to(self.device).view(-1, 1)

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()

            predict_student, f1 = self.model1(x)
            predict_teacher, f2 = self.model2(torch.flip(x, [3]))

            self.loss1 = self.loss_fn(predict_student, y.float().detach())
            self.loss_consistency1 = self.loss_fn(predict_student, Variable(predict_teacher))
            self.sum_loss1 = self.loss1 + self.loss_consistency1
            self.sum_loss1.backward()
            self.optimizer1.step()

            self.loss2 = self.loss_fn(predict_teacher, y.float().detach())
            self.loss_consistency2 = self.loss_fn(predict_teacher, Variable(predict_student))
            self.sum_loss2 = self.loss2 + self.loss_consistency2
            self.sum_loss2.backward()
            self.optimizer2.step()

            self.logger.add_scalar(tag='labeled/supervison',
                                   scalar_value=self.loss1.item(),
                                   global_step=self.global_step)
            self.logger.add_scalar(tag='labeled/consis',
                                   scalar_value=self.loss_consistency1.item(),
                                   global_step=self.global_step)
            self.logger.add_scalar(tag='labeled/sum_loss',
                                   scalar_value=self.sum_loss1.item(),
                                   global_step=self.global_step)
            self.global_step += 1

            if (epoch + 1) == 40:
                model_name1 = '{}-{:0>5d}.pt'.format(self.model_name1, epoch)
                model_name1 = os.path.join(self.ckpt_path, model_name1)
                model_name2 = '{}-{:0>5d}.pt'.format(self.model_name2, epoch)
                model_name2 = os.path.join(self.ckpt_path, model_name2)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model1.state_dict(),
                    'optimizer': self.optimizer1.state_dict(),
                }, model_name1)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model2.state_dict(),
                    'optimizer': self.optimizer2.state_dict(),
                }, model_name2)

    def evl(self):
        y_ = []
        y_pred = []
        self.model1.eval()
        self.model2.eval()
        if self.config.data_mode == 'test':
            with torch.no_grad():
                for index, (images, labels) in enumerate(self.train_dataloader):
                    images = images.cuda()
                    outputs1, flat1 = self.model1(images)
                    outputs2, flat2 = self.model2(torch.flip(images, [3]))
                    outputs = (outputs1 + outputs2) / 2
                    y_pred.extend(outputs.squeeze(dim=1).cpu())
                    y_.extend(labels)

                RMSE, PLCC, SROCC, KROCC = compute_metric(np.array(y_), np.array(y_pred))
                plt.figure(figsize=(6, 6.5))
                plt.scatter(y_, y_pred, s=10)
                plt.xlabel('MOS', fontsize='16', loc='center')
                plt.ylabel('MOS predicted', fontsize='16', loc='center')
                plt.xticks(fontsize='16')
                plt.yticks(fontsize='16')
                plt.xlim((1, 5))
                plt.ylim((1, 5))
                plt.show()
        return PLCC, SROCC

    def _load_checkpoint(self, ckpt1, ckpt2):
        if os.path.isfile(ckpt1):
            print("[*] loading checkpoint '{}'".format(ckpt1))
            checkpoint = torch.load(ckpt1)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model1.load_state_dict(checkpoint['state_dict'])
            self.optimizer1.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer1.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt1, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt1))

        if os.path.isfile(ckpt2):
            print("[*] loading checkpoint '{}'".format(ckpt2))
            checkpoint = torch.load(ckpt2)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model2.load_state_dict(checkpoint['state_dict'])
            self.optimizer2.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer1.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt2, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt2))

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        plcc_, srocc_ = t.evl()
        print(plcc_, srocc_)


if __name__ == "__main__":
    config = parse_config()
    for i in range(5, 6):
        config = parse_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.train_description1)
        main(config)
