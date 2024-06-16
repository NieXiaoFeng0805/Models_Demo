# author: Feng
# contact: 1245272985@qq.com
# datetime:2024/6/16 17:08
# software: PyCharm
"""
文件说明：

"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
import torch
import torch.nn.functional as F
import logging
import numpy as np
from utils.tools import AvgMeter, adjust_lr, clip_gradient, CalParams
from utils.dataset import get_loader, TestDataset
from datetime import datetime
from torch.autograd import Variable

from backbones.U_Net import Unet
from backbones.Polyp_PVT import PolypPVT
from backbones.LDNet.LDNet import LDNet
from backbones.Demo import Demo

cfg = open("./cfg.yaml", encoding='UTF-8')
config = yaml.safe_load(cfg)


def structure_loss(pred, mask):
    """
    结构化损失计算
    :param pred:
    :param mask:
    :return:
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_test(model, path, dataset, size):
    """
    每轮结束后对测试集进行验证观察Dice系数
    :param model: 当前模型
    :param path: 数据集根目录
    :param dataset: 数据集
    :param size: 统一尺寸
    :return:
    """
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = TestDataset(image_root, gt_root, size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        if config['out'][model_name] == 1:
            res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        elif config['out'][model_name] == 2:
            res, res1 = model(image)
            res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        # eval Dice
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
        DSC = DSC + dice

    return DSC / num1


best_loss = 1000.


def train(train_loader, model, optimizer, epoch):
    model.train()
    size_rates = config['Train']['random_scale_crop_range']  # 多尺度训练
    lr = optimizer.state_dict()['param_groups'][0]['lr']  # 当前学习率
    print("current learning rate:", lr)

    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            trainsize = int(round(int(config['Train']['size']) * rate / 32) * 32)
            # 当前rate不为1则将图放大至训练尺寸大小后再进行处理
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # 对于当前模型的输出进行判断
            if config['out'][model_name] == 1:
                out = model(images)
                loss = structure_loss(out, gts)
            elif config['out'][model_name] == 2:
                loss1, loss2 = model(images)
                _loss1 = structure_loss(loss1, gts)
                _loss2 = structure_loss(loss2, gts)
                loss = _loss1 + _loss2

            loss.backward()  # 反向传播更新loss
            # 剪切梯度，防止梯度爆炸或消失，通过剪切掉不合适的梯度将梯度的范围控制在合理的范围内
            clip_gradient(optimizer, float(config['Train']['gradient_clipping_margin']))
            optimizer.step()  # 梯度更新
            if rate == 1:  # 只有在rate==1时才更新loss
                loss_record.update(loss.data, int(config['dataset']['batch_size']))
        # 打印loss
        if i % 50 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' cur_loss: {:0.4f}]'.
                  format(datetime.now(), epoch, int(config['Train']['epoch']), i, total_step,
                         loss_record.show()))
    # 打印当前轮次的loss
    print('Cur Epoch [{:03d}/{:03d}], cur_loss: {:0.4f}]'.
          format(epoch, int(config['Train']['epoch']), loss_record.show()))
    # 保存loss到文件
    loss_path = './Results/{}_loss/'.format(model_name)
    loss_file = loss_path + '{}_loss.txt'.format(model_name)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    with open(loss_file, 'a', encoding='utf-8') as file:
        file.write('Epoch:{}  Loss:{} \n'.format(epoch, loss_record.show()))
        file.close()


    # 测试dice指标
    test_path = './Data/TestDataset/'
    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        dataset_dice = dice_test(model, test_path, dataset, 352)
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
        print(dataset, ': ', dataset_dice)
    logging.info("*" * 30)
    print("第{}轮Dice指标测试结束".format(epoch))

    # 保存权重
    save_path = (config['Train']['checkpoint_save_path'])
    global best_loss
    if epoch > (int(config['Train']['epoch']) // 4) and loss_record.show() < best_loss:
        best_loss = min(loss_record.show(), best_loss)
        torch.save(model.state_dict(), save_path + model_name + '_best' + '.pth')
        print("save {} weight done!".format(epoch))




if __name__ == '__main__':
    # 选择模型
    model_name = config['model']['model_name']['Unet']

    if model_name == 'Unet':
        model = Unet(3, 1).cuda()
    elif model_name == 'PolypPVT':
        model = PolypPVT().cuda()
    elif model_name == 'LDNet':
        model = LDNet().cuda()
    elif model_name == 'Demo':
        model = Demo().cuda()
    print("当前Model为：", model_name)

    # 打印log
    if not os.path.exists(config['Train']['logger_path']):
        os.makedirs(config['Train']['logger_path'])
    logging.basicConfig(filename=config['Train']['logger_path'] + str(model_name) + '.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # 选择优化函数
    if config['Train']['optimizer'] == 'AdamW':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), float(config['Train']['lr']), weight_decay=1e-4, momentum=0.9)

    # 构建训练图像及标签加载器
    img_root = config['dataset']['train_img_root']
    gt_root = config['dataset']['train_label_root']
    train_loader = get_loader(img_root, gt_root, batch_size=int(config['dataset']['batch_size']),
                              train_size=int(config['Train']['size']))
    total_step = len(train_loader)

    # 计算参数
    CalParams(model, torch.randn(2, 3, 352, 352).cuda())

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(1, config['Train']['epoch']):
        adjust_lr(optimizer, float(config['Train']['lr']), epoch, 0.8, 100)
        train(train_loader, model, optimizer, epoch)
