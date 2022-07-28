import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]

def build_bb(in_channels, mid_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,mid_channels,3,dilation=2, padding=2),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels,out_channels,3,dilation=2, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,dilation=2, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

def build_decoder(in_channels, mid_channels_1, mid_channels_2, out_channels, last_bnrelu, upsample_flag):
    layers = []
    layers += [nn.Conv2d(in_channels,mid_channels_1,3,padding=1),
            nn.BatchNorm2d(mid_channels_1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels_1,mid_channels_2,3,padding=1),
            nn.BatchNorm2d(mid_channels_2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels_2,out_channels,3,padding=1)]

    if last_bnrelu:
        layers += [nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),]
    
    if upsample_flag:
        layers += [nn.Upsample(scale_factor=2, mode='bilinear')]

    sequential = nn.Sequential(*layers)
    return sequential


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.resnet = models.resnet34(pretrained=True)
        self.encoder0 = nn.Sequential(
        self.resnet.conv1,
        self.resnet.bn1,
        self.resnet.relu)
        self.encoder1 = nn.Sequential(
        self.resnet.maxpool,
        self.resnet.layer1)
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4
        self.psp_module = PSPModule(512, 512, (1, 3, 5))
        self.psp4 = conv_up_psp(512, 256, 2)
        self.psp3 = conv_up_psp(512, 128, 4)
        self.psp2 = conv_up_psp(512, 64, 8)
        self.psp1 = conv_up_psp(512, 64, 16)

        self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
        self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
        self.decoder1_g = build_decoder(128, 64, 64, 64, True, True)

        self.bridge_block = build_bb(512, 512, 512)
        self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
        self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
        self.decoder1_f = build_decoder(128, 64, 64, 64, True, True)
        

        self.bridge_block = build_bb(512, 512, 512)
        self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_f = build_decoder(256, 256, 256, 128, True, True)
        self.decoder2_f = build_decoder(128, 128, 128, 64, True, True)
        self.decoder1_f = build_decoder(64, 64, 64, 64, True, True)
        self.decoder0_f = build_decoder(64, 64, 64, 1, False, True)

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

        self.pre_focus = nn.Conv2d(256, 512, kernel_size=16, stride=6)

    def forward(self, input):
        
        e0 = self.encoder0(input)
        print(e0.shape)
        e1 = self.encoder1(e0)
        print(e1.shape)
        e2 = self.encoder2(e1)
        print(e2.shape)
        e3 = self.encoder3(e2)
        print(e3.shape)
        e4 = self.encoder4(e3)
        print(e4.shape)
        print("--------------------")

        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)

        glance_sigmoid = F.sigmoid(self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 )))

        output_feature = self.pre_focus(output_feature)

        focus_sigmoid = F.sigmoid(d0_f)

        return glance_sigmoid, focus_sigmoid
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        # self.bridge_block = build_bb(512, 512, 512)
        # self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
        # self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
        # self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
        # self.decoder1_f = build_decoder(128, 64, 64, 64, True, True)
        # self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)

        # self.bridge_block = build_bb(256, 256, 256)
        # self.decoder4_f = build_decoder(512, 256, 128, 2048, True, True)

        # self.decoder3_f = build_decoder(2048*2, 2048, 2048, 1024, True, True)
        # self.decoder2_f = build_decoder(1024*2, 1024, 1024, 512, True, True)
        # self.decoder1_f = build_decoder(512*2, 512, 512, 256, True, True)
        # self.decoder0_f = build_decoder(256*2, 256, 256, 1, False, True)

        self.bridge_block = build_bb(512, 512, 512)
        self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
        self.decoder3_f = build_decoder(256, 256, 256, 128, True, True)
        self.decoder2_f = build_decoder(128, 128, 128, 64, True, True)
        self.decoder1_f = build_decoder(64, 64, 64, 64, True, True)
        self.decoder0_f = build_decoder(64, 64, 64, 1, False, True)

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

        self.pre_focus = nn.Conv2d(256, 512, kernel_size=16, stride=6)

    def forward(self, feature):

        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)

        glance_sigmoid = F.sigmoid(self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 )))

        output_feature = self.pre_focus(output_feature)

        # upscale_size = 112
        # bb = self.bridge_block(output_feature)
        # d4_f = self.decoder4_f(torch.cat((bb, output_feature),1))
        # e3 = F.interpolate(feature['layer4'], size=upscale_size,  mode='bilinear', align_corners=False)
        # print(f"d4_f {d4_f.shape} <> layer4 {e3.shape}")
        # d3_f = self.decoder3_f(torch.cat((d4_f, e3.cuda()),1))
        # e2 = F.interpolate(feature['layer3'], size=upscale_size*2,  mode='bilinear', align_corners=False)
        # print(f"d3_f {d3_f.shape} <> layer3 {e2.shape}")

        # d2_f = self.decoder2_f(torch.cat((d3_f, e2.cuda()),1))
        # e1 = F.interpolate(feature['layer2'], size=upscale_size*4,  mode='bilinear', align_corners=False)
        # print(f"d4_f {d2_f.shape} <> layer2 {e1.shape}")
        # d1_f = self.decoder1_f(torch.cat((d2_f, e1.cuda()),1))
        # e0 = F.interpolate(feature['layer1'], size=upscale_size*8,  mode='bilinear', align_corners=False)
        # print(f"d1_f {d1_f.shape} <> layer1 {e0.shape}")
        # d0_f = self.decoder0_f(torch.cat((d1_f, e0.cuda()),1))
        # print(f"d0_f {d0_f.shape}")


        bb = self.bridge_block(output_feature)
        d4_f = self.decoder4_f(torch.cat((bb, output_feature),1))
        # print(f"d4_f {d4_f.shape}")
        d3_f = self.decoder3_f(d4_f)
        # print(f"d3_f {d3_f.shape}")
        d2_f = self.decoder2_f(d3_f)
        # print(f"d4_f {d2_f.shape}")
        d1_f = self.decoder1_f(d2_f)
        # print(f"d1_f {d1_f.shape}")
        d0_f = self.decoder0_f(d1_f)
        # print(f"d0_f {d0_f.shape}")



        focus_sigmoid = F.sigmoid(d0_f)

        return glance_sigmoid, focus_sigmoid
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module