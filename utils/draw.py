# author: Feng
# contact: 1245272985@qq.com
# datetime:2023/4/4 20:11
# software: PyCharm
"""
文件说明：

"""
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import glob
from PIL import Image
import cv2

cfg = open("../cfg.yaml", encoding='UTF-8')
config = yaml.safe_load(cfg)


def draw_loss(loss_file_list, model_name_list):
    # 打开loss文件并将其添加到数组中
    lens = len(loss_file_list)
    epoch_list = [[] for _ in range(lens)]
    loss_list = [[] for _ in range(lens)]
    for i in range(lens):
        loss_file = loss_file_list[i]
        model_name = model_name_list[i]
        with open(loss_file, 'r') as file:
            for line in file.readlines():
                line = line.strip().split(" ")
                epoch, loss = line[0], line[2]
                epoch = int(epoch.split(':')[1])
                epoch_list[i].append(epoch)
                loss = float(loss.split(':')[1])
                loss_list[i].append(loss)
        plt.title('Struct Loss')
        plt.plot(epoch_list[i], loss_list[i], label='Loss of {}'.format(model_name))

    plt.xlabel('Epoch')
    plt.ylabel('Loss_item')
    plt.legend()
    plt.savefig('../Results/loss.png', dpi=1000)
    print("Loss Figure Done")


def merge_img(model_name_list):
    for model in model_name_list:
        for dataset_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            ori_root = config['dataset']['test_' + str(dataset_name) + '_img']
            gt_root = config['dataset']['test_' + str(dataset_name) + '_label']
            pred_root = '../Results/{}/{}/'.format(model, dataset_name)

            ori = [ori_root + f for f in os.listdir(ori_root) if f.endswith('.jpg') or f.endswith('.png')]
            gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
            pred = [pred_root + f for f in os.listdir(pred_root) if f.endswith('.jpg') or f.endswith('.png')]

            save_path = '../Results/Concat_result/{}/'.format(dataset_name)
            nums = len(ori)
            for i in range(nums):
                img = cv2.imread(ori[i])
                pre = cv2.imread(pred[i])
                ground_turth = cv2.imread(gt[i])
                out = cv2.hconcat([img, pre, ground_turth])
                # cv2.imshow("Result", out)
                cv2.imwrite(save_path + 'result' + str(i) + '.png', out)


if __name__ == '__main__':
    draw_loss(
        ['../log/CTDPN_loss.txt', '../log/DMFN_loss.txt', '../log/FBCAN_loss.txt', '../log/SSFormer_loss.txt',
         '../log/MyModel_loss.txt', '../log/MyModel_v1_loss.txt', '../log/PolypPVT_loss.txt'],
        ['CTDPN', 'DMFN', 'FBCAN', 'SSFormer', 'MyModel', 'MyModel_v1', 'PolypPVT'])
    # merge_img([config['model']['model_name']['Unet'], config['model']['model_name']['MyModel']])
