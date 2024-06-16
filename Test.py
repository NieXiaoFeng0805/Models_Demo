# author: Feng
# contact: 1245272985@qq.com
# datetime:2024/6/16 17:46
# software: PyCharm
"""
文件说明：

"""
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn.functional as F
import numpy as np
import cv2
import yaml
from utils.tools import CalParams
# from utils.Draw import draw_featuremap
from utils.dataset import TestDataset


cfg = open("./cfg.yaml", encoding='UTF-8')
config = yaml.safe_load(cfg)


def test():
    from Best_version import CTDPN
    choice_model = "Best_version"
    model = CTDPN().cuda().eval()
    weight_path = './checkpoint/FinalVersion_5_62.pth'
    # weight_path = './checkpoint/Best_version.pth'
    model.load_state_dict(torch.load(weight_path), strict=False)
    # 计算参数
    CalParams(model, torch.randn(2, 3, 352, 352).cuda())
    print("当前预测模型：{},选用权重：{}".format(choice_model, weight_path))
    # 开始测试
    for dataset_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        print(f" predicting {dataset_name}")
        img_root = config['dataset']['test_' + str(dataset_name) + '_img']
        gt_root = config['dataset']['test_' + str(dataset_name) + '_label']
        save_path = './Results/{}/{}/'.format(choice_model, dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_feature = './Results/FeatureMaps/{}/{}/'.format(choice_model, dataset_name)
        if not os.path.exists(save_feature):
            os.makedirs(save_feature)
        test_loader = TestDataset(img_root, gt_root, int(config['Train']['size']))

        len_of_data = len(os.listdir(gt_root))
        dice_list = []
        for i in range(len_of_data):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            if config['out'][choice_model] == 4:
                P1, P2, P3, P4 = model(image)
                res = F.upsample(P1 + P2 + P3 + P4, size=gt.shape, mode='bilinear', align_corners=False)
                # # 看看featuremap
                # heatmap = F.upsample(P4, size=gt.shape, mode='bilinear', align_corners=False)
                # draw_featuremap(heatmap, name, save_feature)

            elif config['out'][choice_model] == 5:
                P1, P2, P3, P4, P5 = model(image)
                res = F.upsample(P1 + P2 + P3 + P4 + P5, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            dice_list.append(dice)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            name = name.split('\\')[-1]
            cv2.imwrite(save_path + name, res * 255)

        print(np.mean(dice_list), dataset_name, 'Finish!')
        # print(dice_list)
    print("Best Modeles Prediction Had Done")

if __name__ == '__main__':
    test()
