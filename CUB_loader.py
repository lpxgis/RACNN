# crop size 448
# mean 109.973, 127.336, 123.883
import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class CUB200_loader(data.Dataset):
    def __init__(self, root):

        std = 1. / 255.
        means = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
        self.transform=None
        self._imgpath = []
        self._imglabel = []
        self.transform = None
        self.files = os.listdir(root+'/all')
        for i in self.files:
            img_path = os.path.join(root+'/all', i)
            self._imgpath.append(img_path)
            self._imglabel.append(0)
        self.files = os.listdir(root + '/part')
        for i in self.files:
            img_path = os.path.join(root + '/part', i)
            self._imgpath.append(img_path)
            self._imglabel.append(1)

        self.transform=transforms.Compose([
                transforms.Resize(448),
                transforms.RandomRotation(45),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = means,
                    std = [std]*3)
                ])


    def __getitem__(self, index):

        img = Image.open(self._imgpath[index])
        img = self.transform(img)
        cls = self._imglabel[index]
        return img, cls

    def __len__(self):
        return len(self._imglabel)

    def idx_to_classname(self, idx):
        return self.idx2name[idx]

    def CUB_collate(self, batch):
        imgs = []
        cls = []
        for sample in batch:
            imgs.append(sample[0])
            cls.append(sample[1])
        imgs = torch.stack(imgs, 0)
        cls = torch.LongTensor(cls)
        return imgs, cls
    @staticmethod
    def tensor_to_img(x, imtype=np.uint8):
        """"将tensor的数据类型转成numpy类型，并反归一化.
        Parameters:
            input_image (tensor) --  输入的图像tensor数组
            imtype (type)        --  转换后的numpy的数据类型
        """
        mean = [109.97 / 255., 127.34 / 255., 123.88 / 255.]
        std = [1. / 255., 1. / 255., 1. / 255.]

        if not isinstance(x, np.ndarray):
            if isinstance(x, torch.Tensor):  # get the data from a variable
                image_tensor = x.data
            else:
                return x
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
            image_numpy = image_numpy * 255
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = x
        return image_numpy.astype(imtype)

if __name__ == '__main__':
    trainset = CUB200_loader(os.getcwd() + '/data/images')
    trainloader = data.DataLoader(trainset, batch_size = 3,
            shuffle = False, collate_fn = trainset.CUB_collate, num_workers = 1)
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        break

    testset = CUB200_loader(os.getcwd() + '/data/images')
    testloader = data.DataLoader(testset, batch_size = 3,
            shuffle = False, collate_fn = testset.CUB_collate, num_workers = 1)

    for img, cls in trainloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break
