import os
import csv
import numpy as np
from numpy import resize
import PIL.Image as Image
import matplotlib.pyplot as plt
import yaml


def Toresize(pred, mask):
    if pred.shape != mask.shape:
        wp, hp = pred.shape
        wm, hm = mask.shape
        if wp < wm or hp < hm:
            h = max(hp, hm)
            w = max(wp, wm)
            return pred.resize((h, w), Image.BILINEAR), mask.resize((h, w), Image.NEAREST)
    return pred, mask


def dice(input, target):
    smooth = 0.001
    input_flat = np.reshape(input, (-1))
    target_flat = np.reshape(target, (-1))
    intersection = (input_flat * target_flat)
    dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
    dice = '{:.4f}'.format(dice)
    dice = float(dice)
    return dice


def model_dice(dir_label, dir_mask, Thresholds):
    dir_label = dir_label
    dir_mask = dir_mask
    list_mask = os.listdir(dir_mask)
    list_label = os.listdir(dir_label)
    num = len(list_mask)
    dice_list100 = np.zeros((num, 10))
    # print(list_mask)
    for n, p in enumerate(list_label):
        label = np.array(Image.open(os.path.join(dir_label, p)).convert('L'))
        label = label / 255
        mask = np.array(Image.open(os.path.join(dir_mask, list_mask[n])))
        mask = mask / 255
        if label.shape != mask.shape:
            label, mask = Toresize(label, mask)
        print(n, p, list_label[n], mask.shape, label.shape)
        for i, th in enumerate(Thresholds):
            Bi_pred = np.zeros_like(mask)
            Bi_pred[mask > th] = 1
            Bi_pred[mask < th] = 0
            d = dice(Bi_pred, label)
            dice_list100[n][i] = d

    return list(np.mean(dice_list100, 0))


if __name__ == "__main__":
    cfg = open("../cfg.yaml", encoding='UTF-8')
    config = yaml.safe_load(cfg)
    Thresholds = np.linspace(0.99, 0, 10)[::-1]  # 分段计算阈值
    dataset = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    model = ['Unet', 'MyModel', 'MyModel_v1', 'PolypPVT']
    all_list = []
    color_list = ['red', 'black', 'aqua', 'blue', 'bisque', 'burlywood', 'darkblue', 'darkgreen', 'chocolate', 'brown',
                  'coral', 'antiquewhite']
    # 路径列表
    dir_label = []

    for j, d in enumerate(dataset):
        plt.title(d)
        index = 0
        for i, m in enumerate(model):
            index = i
            path = os.path.join(r'../Results', m, d)
            dir_label = '../' + config['dataset']['test_' + d + '_label']
            # path = os.path.join(r'../Results', m, 'Kvasir')
            # dir_label = '../' + config['dataset']['test_Kvasir_label']
            print("当前model为：", m)
            outs = model_dice(dir_label, path, Thresholds)
            plt.plot(Thresholds, outs, color=color_list[i], label=m)
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Dice')
        plt.savefig('../Results/Dice Of Every Models In {}.jpg'.format(d), dpi=750)
        plt.close()
        print("{}下各模型的Dice测试结果完成".format(dataset[j]))
