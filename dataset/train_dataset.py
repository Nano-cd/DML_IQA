from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision

data_transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize([512, 384]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


class train_data(Dataset):
    def __init__(self, label_file, image_root):
        self.imagepaths = []
        self.labels = []
        self.labeled_index = []
        self.transforms = data_transform_train

        with open(label_file, 'r') as f:
            for line in f.readlines():  # 读取label文件
                self.imagepaths.append(os.path.join(image_root, line.split()[1]))
                self.labels.append(float(line.split()[0]))

    def __getitem__(self, item):
        x = Image.open(self.imagepaths[item])
        y = self.labels[item]
        return self.transforms(x), y

    def __len__(self):
        return len(self.imagepaths)
