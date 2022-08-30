import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import pytorch_lightning
import numpy as np

def collaborative_matting_(glance_sigmoid, focus_sigmoid):
    values, index = torch.max(glance_sigmoid,1)
    index = index[:,None,:,:].float()
    ### index <===> [0, 1, 2]
    ### bg_mask <===> [1, 0, 0]
    bg_mask = index.clone()
    bg_mask[bg_mask==2]=1
    bg_mask = 1- bg_mask
    ### trimap_mask <===> [0, 1, 0]
    trimap_mask = index.clone()
    trimap_mask[trimap_mask==2]=0
    ### fg_mask <===> [0, 0, 1]
    fg_mask = index.clone()
    fg_mask[fg_mask==1]=0
    fg_mask[fg_mask==2]=1
    focus_sigmoid = focus_sigmoid.cpu()
    trimap_mask = trimap_mask.cpu()
    fg_mask = fg_mask.cpu()
    fusion_sigmoid = focus_sigmoid*trimap_mask+fg_mask

    fusion_sigmoid = fusion_sigmoid.cuda()
    return fusion_sigmoid

def get_masked_local_from_global_test(global_result, local_result):
	weighted_global = np.ones(global_result.shape)
	weighted_global[global_result==255] = 0
	weighted_global[global_result==0] = 0
	fusion_result = global_result*(1.-weighted_global)/255+local_result*weighted_global
	return fusion_result
	
def gen_trimap_from_segmap_e2e(segmap):
	trimap = np.argmax(segmap, axis=1)[0]
	trimap = trimap.astype(np.int64)	
	trimap[trimap==1]=128
	trimap[trimap==2]=255
	return trimap.astype(np.uint8)

def gen_bw_from_segmap_e2e(segmap):
	bw = np.argmax(segmap, axis=1)[0]
	bw = bw.astype(np.int64)
	bw[bw==1]=255
	return bw.astype(np.uint8)

def save_test_result(save_dir, predict):
	predict = (predict * 255).astype(np.uint8)
	cv2.imwrite(save_dir, predict)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_up_psp(in_channels, out_channels, up_sample):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=up_sample, mode='bilinear',align_corners = False))

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

    if last_bnrelu:
        layers += [nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),]
    
    if upsample_flag:
        layers += [nn.Upsample(scale_factor=2, mode='bilinear')]

    sequential = nn.Sequential(*layers)
    return sequential

class BasicBlock(pytorch_lightning.LightningModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class PSPModule(pytorch_lightning.LightningModule):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners = True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class SELayer(pytorch_lightning.LightningModule):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DeepLabV3Plus(pytorch_lightning.LightningModule):
    def __init__(self, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(64, 48, 1, bias=False),
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

        self.decoder0_g = build_decoder(128, 64, 64, 3, False, True)
        self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)

        #
        # Need to replace this with e-ASPP
        #
        self.aspp = ASPP(512, aspp_dilate)

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
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        low_level_feature = self.project(e1)
        output_feature = self.aspp(e4)
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        glance_sigmoid = F.sigmoid(self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 )))
        input_shape = input.shape[-2:]
        glance_sigmoid = F.interpolate(glance_sigmoid, size=input_shape, mode='bilinear', align_corners=False)

        # output_feature = self.pre_focus(output_feature)

        bb = self.bridge_block(e4)
        d4_f = self.decoder4_f(torch.cat((bb, e4),1))
        d3_f = self.decoder3_f(torch.cat((d4_f, e3),1))    
        d2_f = self.decoder2_f(torch.cat((d3_f, e2),1))
        d1_f = self.decoder1_f(torch.cat((d2_f, e1),1))
        d0_f = self.decoder0_f(torch.cat((d1_f, e0),1))
        focus_sigmoid = F.sigmoid(d0_f)

        fusion_sigmoid = collaborative_matting_(glance_sigmoid, focus_sigmoid)

        return glance_sigmoid, focus_sigmoid, fusion_sigmoid
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(pytorch_lightning.LightningModule):
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

class ASPP(pytorch_lightning.LightningModule):
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