# author: Feng
# contact: 1245272985@qq.com
# datetime:2024/6/16 21:14
# software: PyCharm
"""
t1缩小两倍，t2，t3，t4复用代替gap，RFAB convcat后接gcblock；全GELU；bs=24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.pvt.pvt_v2 import pvt_v2_b2
from utils.tools import CalParams


# 深度可分离卷积
class DWconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(0, 0), dilation=1):
        super(DWconv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# 基本卷积
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, gelu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.gelu = nn.GELU() if gelu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.gelu = nn.GELU() if gelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.gelu is not None:
            x = self.gelu(x)
        return x


# CAB 交叉注意模块
class cabChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=30):
        super(cabChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            cabChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


# CBAM模块 通道+空间串联
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.gelu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.gelu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ASPP_Module(nn.Module):
    def __init__(self, in_channel, out_channel, output_strid):
        '''
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param output_strid: 控制不同的膨胀率
        '''
        super(ASPP_Module, self).__init__()
        dilations = []
        if output_strid == 16:
            dilations = [1, 6, 12, 18]
        elif output_strid == 8:
            dilations = [1, 12, 24, 36]

        # 因为是并行的，所以所有的块的输入通道和输出通道都是相同的
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3],
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU()
        )
        # 将前面的输出进行concat 5*128；（concat在前向传播中进行实现）
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 初始化卷积核(可有可无)
        # self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        # print('x1 size = ', x1.size())
        x2 = self.aspp2(x)
        # print('x2 size = ', x2.size())
        x3 = self.aspp3(x)
        # print('x3 size = ', x3.size())
        x4 = self.aspp4(x)
        # print('x4 size = ', x4.size())

        # 进行拼接
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # print('拼接后的维度', x.size())
        # 进行1×1 的卷积和归一化处理
        x = self.conv1(x)
        # print('经过1×1卷积后的维度', x.size())
        x = self.bn1(x)
        return x


# 多分支特征聚合模块
class BFAM(nn.Module):
    '''Branch feature aggregation module'''

    def __init__(self, in_channel, out_channel):
        super(BFAM, self).__init__()
        self.gelu = nn.GELU()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            DWconv(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            DWconv(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            DWconv(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            DWconv(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            DWconv(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            DWconv(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            DWconv(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            DWconv(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            DWconv(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            CAB(out_channel)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
        )  # 8.20 19:52改
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
        self.gc = GlobalContextBlock(out_channel)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_cat = self.gc(x_cat) + x_cat

        x = self.gelu(x_cat * self.branch4(x) + self.conv_res(x))
        return x


def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channel=128, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)  # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)  # bs,k,c
        bar_a = self.softmax(hat_a)  # bs,k,c
        attention = bar_a.unsqueeze(-2)  # #bs,k,1,c
        out = attention * x_all  # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=128):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = value
        return out


class FGC(nn.Module):
    def __init__(self, dims, dim):
        super(FGC, self).__init__()
        t2_dim, t3_dim, t4_dim = dims
        self.pro_2 = nn.Conv2d(t2_dim, dim, kernel_size=1)
        self.pro_3 = nn.Conv2d(t3_dim, dim, kernel_size=1)
        self.pro_4 = nn.Conv2d(t4_dim, dim, kernel_size=1)
        self.fgc_t2 = GlobalContextBlock(dim)
        self.fgc_t3 = GlobalContextBlock(dim)
        self.fgc_t4 = GlobalContextBlock(dim)

        self.fgc_out = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            GlobalContextBlock(dim),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, t2, t3, t4 = x
        t2 = self.pro_2(t2)
        t3 = self.pro_3(t3)
        t4 = self.pro_4(t4)
        t2 = self.fgc_t2(t2) * t2
        t3 = self.fgc_t3(t3) * t3
        t4 = self.fgc_t4(t4) * t4
        x = torch.cat((t2, self.up2(t3), self.up4(t4)), dim=1)
        x = self.fgc_out(x)
        return x


class Demo(nn.Module):
    def __init__(self, f_dims, dim):
        super(Demo, self).__init__()
        t1_dim, t2_dim, t3_dim, t4_dim = f_dims
        # 投影
        self.pro_t2 = nn.Sequential(
            nn.Conv2d(t2_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )
        self.pro_t3 = nn.Sequential(
            nn.Conv2d(t3_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )
        self.pro_t4 = nn.Sequential(
            nn.Conv2d(t4_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim), )
        self.pro_out = nn.Sequential(
            BasicConv(dim // 4, dim // 8, kernel_size=3, padding=1),
            nn.Conv2d(dim // 8, 1, kernel_size=1)
        )

        # BFAM,卷积
        self.bfam_1 = BFAM(dim * 2, dim // 4)
        self.bfam_2 = BFAM((dim + t1_dim), dim // 4)
        self.aspp = ASPP_Module(dim - dim // 4, dim // 4, output_strid=16)
        # 变换尺度
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # Attention
        self.s2att = S2Attention(dim // 2)
        self.ca = ChannelAttention(dim // 4)
        self.sa = SpatialAttention(kernel_size=7)
        # 激活
        self.gelu = nn.GELU()

    def forward(self, x, fgc):
        t1, t2, t3, t4 = x
        # 投影
        t2 = self.pro_t2(t2)
        t3 = self.pro_t3(t3)
        t4 = self.pro_t4(t4)
        # t1 branch
        t1 = self.ca(t1) * t1
        t1 = self.sa(t1) * t1
        t1 = F.interpolate(t1, scale_factor=0.5, mode='bilinear', align_corners=True)
        # 分支聚合
        t2 = self.gelu(fgc * t2 + t2)
        t3 = self.gelu(fgc * t3 + t3)
        bfam_34 = self.bfam_1(torch.cat((t3, self.up2(t4)), dim=1))
        bfam_24 = self.bfam_1(torch.cat((t2, self.up4(t4)), dim=1))
        bfam_14 = self.bfam_2(torch.cat((t1, self.up4(t4)), dim=1))

        # out
        bfam_234 = self.s2att(torch.cat((bfam_24, self.up2(bfam_34)), dim=1))
        bfam_out = torch.cat((bfam_234, bfam_14), dim=1)
        out = self.aspp(bfam_out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.pro_out(out)
        bfam1_out = self.pro_out(bfam_14)
        bfam2_out = self.pro_out(bfam_24)
        bfam3_out = self.pro_out(bfam_34)

        return out, bfam1_out, bfam2_out, bfam3_out


class CTDPN(nn.Module):
    def __init__(self, class_num=1):
        super(CTDPN, self).__init__()
        self.class_num = class_num
        self.backbone = pvt_v2_b2()
        self.FGC = FGC([128, 320, 512], 256)
        self.CTDPN = Demo([64, 128, 320, 512], 256)
        self.init_weights()

    def init_weights(self):
        # 加载预训练权重
        pretrain_path = "PreWeights/pvt_v2_b2.pth"
        save_model = torch.load(pretrain_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        print("Success load pre weight!")

    def forward(self, x):
        outs = self.backbone(x)
        fgc = self.FGC(outs)
        P1, P2, P3, P4 = self.CTDPN(outs, fgc)
        P1 = F.interpolate(P1, scale_factor=8, mode='bilinear', align_corners=True)
        P2 = F.interpolate(P2, scale_factor=8, mode='bilinear', align_corners=True)
        P3 = F.interpolate(P3, scale_factor=8, mode='bilinear', align_corners=True)
        P4 = F.interpolate(P4, scale_factor=16, mode='bilinear', align_corners=True)
        return P1, P2, P3, P4


if __name__ == '__main__':
    model = CTDPN().cuda()
    input = torch.randn(2, 3, 352, 352).cuda()
    # 计算参数
    CalParams(model, input)
    P1, P2, P3, P4 = model(input)
    print(P1.shape, P2.shape, P3.shape, P4.shape)
