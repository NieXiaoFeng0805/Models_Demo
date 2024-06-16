# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/4/3 21:18
# software: PyCharm
"""
文件说明：

"""
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


class TheDataset(data.Dataset):
    """
    data pre-processing file
    """

    def __init__(self, image_root, gt_root, train_size):
        """
        init file
        :param image_root: Path of images
        :param gt_root: Path of GT
        :param train_size: Size of Images
        """
        self.train_size = train_size
        # Get images and gts path
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('jpg') or f.endswith('png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # Call filter function of files(images)
        self.filter_file()
        # Get images length
        self.size = len(self.images)
        # Define a transform to do the normalization of the data
        print('Using RandomRotation, RandomFlip')
        self.img_transform = transforms.Compose([
            # transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
            transforms.RandomVerticalFlip(p=0.5),  # 一半的照片被增强了！
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            # transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        """
        get the items
        :param index:index of images and gts
        :return: images and gts
        """
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        image = self.img_transform(image)
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        gt = self.gt_transform(gt)
        return image, gt

    def filter_file(self):
        """
        Add the paths for image and Gt to the list
        :return:
        """
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        """
        Specifies to convert to 8 pixels in black and white mode
        :param path:Path of images
        :return:
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """
        Convert GT to binary encoding
        :param path:Path of GTs
        :return:
        """
        with open(path, 'rb') as f:
            gt = Image.open(f)
            return gt.convert('L')

    def resize(self, img, gt):
        """
        Change size for img and gt
        :param img:
        :param gt:
        :return:
        """
        assert img.size == gt.size
        w, h = img.size
        if h < self.train_size or w < self.train_size:
            h = max(h, self.train_size)
            w = max(w, self.train_size)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batch_size, train_size, shuffle=False, num_worker=4, pin_memory=True):
    """
    Call the data preprocessor class to get the combed data
    :param image_root:
    :param gt_root:
    :param batch_size:
    :param train_size:
    :param shuffle: Sort the elements randomly
    :param num_worker:Process number allocation
    :param pin_memory:锁页内存设置(内存足True，不足False)
    :return:
    """
    dataset = TheDataset(image_root, gt_root, train_size)
    # print(dataset[0], dataset[1])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_worker,
                                  pin_memory=pin_memory)
    return data_loader


class TestDataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # print("1", image.size)
        # image = blur_demo(image)# 模糊
        image = self.transform(image).unsqueeze(0)
        # print("1", image.shape)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':
    import cv2 as cv

    train_loader = get_loader('../Data/TrainDataset/image/', '../data/TrainDataset/mask/', batch_size=1,
                              train_size=224)

    for i, pack in enumerate(train_loader, start=1):
        # print(i, pack)
        img, gt = pack[0].squeeze(0), pack[1].squeeze(0)
        print(img.shape)
        # img = img.numpy()
        # img = np.uint8(img).transpose(1, 2, 0)
        # print(img.shape)
        # cv.imshow('img',img)
        plt.imshow(transforms.ToPILImage()(img), interpolation='bicubic')
        plt.imshow(transforms.ToPILImage()(gt), interpolation='bicubic')

        transforms.ToPILImage()(img).show()
        transforms.ToPILImage()(gt).show()
