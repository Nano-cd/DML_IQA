import argparse
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter as sum_writer
import dataset.create_dataset
import dataset.MB_dataset
import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn
import matplotlib.pyplot as plt
import utils
from model_hub import Alexnet_modify, VGG16_modify, resnet34_modify, mobilenet_v2_modify, Inceptionresnetv2_modify, \
    densenet121_modify, Inceptionresnetv2_modify, MB_CNN_modify
from utils import compute_metric


def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=str, default='normal')

    parser.add_argument("--network", type=str, default='mbcnn'
                        , help='resnet34, vgg16, '
                               'densenet121, mobilenet_v2, alexnet, Inceptionresnetv2')

    parser.add_argument("--data_mode", type=str, default='train', help='train test')
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument('--data', type=str, default='MB')
    parser.add_argument("--train_description", type=str, default='MB_BL8K',
                        help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--split', type=int, default='1')

    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='Resnet34BL8K-00040.pt',
                        type=str, help='name of the checkpoint to load')
    parser.add_argument('--tensorboard_path', default='./runs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--number_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--decay_interval", type=int, default=50)
    parser.add_argument("--decay_ratio", type=float, default=0.9)
    parser.add_argument("--init", type=str, default='kaiming_norm')

    return parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
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
        elif self.config.data == 'MB':
            self.train_data = dataset.MB_dataset.IQADataset(
                './data/kon10k/train8000/train_' + str(config.split),
                './data/kon10k/1024x768',
                './gradiant_data/1024x768')
            self.train_dataloader = DataLoader(self.train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=config.number_workers,
                                               pin_memory=config.pin_memory)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the model
        if config.network == 'resnet34':
            print('resnet34 model selected')
            self.model = resnet34_modify.Resnet34(num_classes=1, init_mode=self.config.init)
        elif config.network == 'vgg16':
            print('vgg16 model selected')
            self.model = VGG16_modify.VGG16(num_classes=1, init_mode=self.config.init)
        elif config.network == 'Inceptionresnetv2':
            print('Inceptionresnetv2 model selected')
            self.model = Inceptionresnetv2_modify.Inceptionresnet_v2(num_classes=1, init_mode=self.config.init)
        elif config.network == 'densenet121':
            print('densenet121 model selected')
            self.model = densenet121_modify.Dense_net121(num_classes=1, init_mode=self.config.init)
        elif config.network == 'alexnet':
            print('alexnet model selected')
            self.model = Alexnet_modify.Alex_net(num_classes=1, init_mode=self.config.init)
        elif config.network == 'mobilenet_v2':
            print('mobilenet_v2 model selected')
            self.model = mobilenet_v2_modify.Mobilenet_v2(num_classes=1, init_mode=self.config.init)
        elif config.network == 'mbcnn':
            print('mbcnn model selected')
            self.model = MB_CNN_modify.CNNIQAnet()
        else:
            raise NotImplementedError("Not supported network, need to be added!")

        self.model.to(self.device)
        self.model_name = type(self.model).__name__ + self.config.train_description

        # initialize the loss function and optimizer
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.loss_fn = torch.nn.L1Loss()
        self.ckpt_path = config.ckpt_path
        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         last_epoch=self.start_epoch - 1,
                                                         step_size=config.decay_interval,
                                                         gamma=config.decay_ratio)
        self.global_step = 1
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name + str(self.config.split))
        self.logger = sum_writer(runs_path)

        if not config.train:
            ckpt = os.path.join(config.ckpt_path, config.ckpt)
            self._load_checkpoint(ckpt=ckpt)

    def fit(self):
        if self.train_mode == 'normal':
            for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
                self._train_one_epoch(epoch)
                self.scheduler.step()

    def _train_one_epoch(self, epoch):
        # start training
        # print('Adam learning rate: {:.8f}'.format(self.optimizer.param_groups[0]['lr']))
        self.model.train()
        for _, data in enumerate(self.train_dataloader):
            x = data[0]
            y = data[1].to(self.device)
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
                x[i] = x[i].to(torch.float32)

            self.optimizer.zero_grad()
            predict_student = self.model(x)
            self.loss = self.loss_fn(predict_student, y.detach())
            self.loss.backward()
            self.optimizer.step()

            self.logger.add_scalar(tag='sum_loss',
                                   scalar_value=self.loss.item(),
                                   global_step=self.global_step)
            self.global_step += 1
            if (epoch + 1) == self.config.max_epoch:
                model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch + 1)
                model_name = os.path.join(self.ckpt_path, model_name)
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, model_name)

    def evl(self):
        y_ = []
        y_pred = []
        self.model.eval()
        if self.config.data_mode == 'test':
            with torch.no_grad():
                for index, (images, labels) in enumerate(self.train_dataloader):
                    images = images.cuda()
                    outputs, fm = self.model(images)
                    y_.extend(labels)
                    y_pred.extend(outputs.squeeze(dim=1).cpu())

                plt.hist2d(np.array(y_), np.abs(np.array(y_pred) - np.array(y_)), bins=50)
                plt.show()
                plt.scatter(np.array(y_), np.array(y_pred))
                plt.show()
                RMSE, PLCC, SROCC, KROCC = compute_metric(np.array(y_), np.array(y_pred))
        return PLCC, SROCC

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

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
    # seed_torch(config)
    for i in range(0, 5):
        config = parse_config()
        split = i + 1
        config.split = split
        config.ckpt_path = os.path.join(config.ckpt_path, str(config.split))
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)
        print(config.network, config.train_description)
        main(config)
