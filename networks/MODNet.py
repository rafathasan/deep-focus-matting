import pytorch_lightning
import scipy
from scipy import ndimage
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from .backbones import SUPPORTED_BACKBONES

from networks.Wrapper import LightningWrapper

class MODNet(LightningWrapper):
    def __init__(
        self,
        settings = None
        ):
        super().__init__(settings)
        self.net = _MODNet(backbone_pretrained=False)
        self.blurer = _GaussianBlurLayer(1, 3)

    def forward(self, x):
        return self.net(x, False)

    def training_step(self, batch, batch_idx):
        image, mask, trimap, fg, bg = batch
        image = image/255
        mask = mask/255
        trimap = trimap/255
        fg = fg/(255*255)
        bg = bg/255

        pred_semantic, pred_detail, pred_matte = self(image)

        # print(f"{image.unique().max()=}")
        # print(f"{mask.unique().max()=}")
        # print(f"{trimap.unique().max()=}")
        # print(f"{fg.unique().max()=}")
        # print(f"{bg.unique().max()=}")
        # print(f"{pred_semantic.unique().max()=}")
        # print(f"{pred_detail.unique().max()=}")
        # print(f"{pred_matte.unique().max()=}")
        
        # calculate the boundary mask from the trimap
        boundaries = (trimap < 0.5) + (trimap > 0.5)
        semantic_scale=10.0
        detail_scale=10.0
        matte_scale=1.0

        # calculate the semantic loss
        gt_semantic = F.interpolate(mask, scale_factor=1/16, mode='bilinear', align_corners=False)
        gt_semantic = self.blurer(gt_semantic) # Added sigmoid to avoid incorrect loss calc
        semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
        semantic_loss = semantic_scale * semantic_loss

        # calculate the detail loss
        pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
        gt_detail = torch.where(boundaries, trimap, mask)
        detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
        detail_loss = detail_scale * detail_loss



        # calculate the matte loss
        pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
        matte_l1_loss = F.l1_loss(pred_matte, mask) + 4.0 * F.l1_loss(pred_boundary_matte, mask)
        matte_compositional_loss = F.l1_loss(image * pred_matte, image * mask) \
            + 4.0 * F.l1_loss(image * pred_boundary_matte, image * mask)
        matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
        matte_loss = matte_scale * matte_loss

        # calculate the final loss
        loss = semantic_loss + detail_loss + matte_loss

        mask = mask*255
        
        if batch_idx == 0:
            self.log_image(title="training_images", predict=pred_matte, mask=mask)
            self.logger.experiment.add_image("training_global",  torchvision.utils.make_grid(F.interpolate(pred_semantic, scale_factor=16, mode='bilinear', align_corners=False)), self.current_epoch)
            self.log_image(title="training_local", predict=pred_detail, mask=mask)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=pred_matte, mask=mask),
            "l2loss": self.l2loss(predict=pred_matte, mask=mask),
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg = batch
            image = image/255
            mask = mask/255
            trimap = trimap/255
            fg = fg/255
            bg = bg/255

            pred_semantic, pred_detail, pred_matte = self(image)
            
            # calculate the boundary mask from the trimap
            boundaries = (trimap < 0.5) + (trimap > 0.5)
            semantic_scale=10.0
            detail_scale=10.0
            matte_scale=1.0

            # calculate the semantic loss
            gt_semantic = F.interpolate(mask, scale_factor=1/16, mode='bilinear', align_corners=False)
            gt_semantic = self.blurer(gt_semantic)
            semantic_loss = torch.mean(F.mse_loss(pred_semantic, gt_semantic))
            semantic_loss = semantic_scale * semantic_loss

            # calculate the detail loss
            pred_boundary_detail = torch.where(boundaries, trimap, pred_detail)
            gt_detail = torch.where(boundaries, trimap, mask)
            detail_loss = torch.mean(F.l1_loss(pred_boundary_detail, gt_detail))
            detail_loss = detail_scale * detail_loss

            # calculate the matte loss
            pred_boundary_matte = torch.where(boundaries, trimap, pred_matte)
            matte_l1_loss = F.l1_loss(pred_matte, mask) + 4.0 * F.l1_loss(pred_boundary_matte, mask)
            matte_compositional_loss = F.l1_loss(image * pred_matte, image * mask) \
                + 4.0 * F.l1_loss(image * pred_boundary_matte, image * mask)
            matte_loss = torch.mean(matte_l1_loss + matte_compositional_loss)
            matte_loss = matte_scale * matte_loss

            # calculate the final loss, backward the loss, and update the model 
            loss = semantic_loss + detail_loss + matte_loss
        
            mask = mask*255

            if batch_idx == 0:
                self.log_image(title="validation_images", predict=pred_matte, mask=mask)
                self.logger.experiment.add_image("validation_global", torchvision.utils.make_grid(pred_semantic), self.current_epoch)
                self.log_image(title="validation_local", predict=pred_detail, mask=mask)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=matte_loss, mask=mask),
            "l2loss": self.l2loss(predict=matte_loss, mask=mask),
        }

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg = batch
            image = image/255
            mask = mask/255
            trimap = trimap/255
            fg = fg/255
            bg = bg/255

            pred_semantic, pred_detail, pred_matte = self(image)

            return pred_matte


#------------------------------------------------------------------------------
#  _MODNet Basic Modules
#------------------------------------------------------------------------------

class _IBNorm(pytorch_lightning.LightningModule):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(_IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class _Conv2d_IBNormRelu(pytorch_lightning.LightningModule):
    """ Convolution + _IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(_Conv2d_IBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(_IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class _SEBlock(pytorch_lightning.LightningModule):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(_SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

#------------------------------------------------------------------------------
#  _MODNet Branches
#------------------------------------------------------------------------------

class _LRBranch(pytorch_lightning.LightningModule):
    """ Low Resolution Branch of _MODNet
    """

    def __init__(self, backbone):
        super(_LRBranch, self).__init__()

        enc_channels = backbone.enc_channels
        
        self.backbone = backbone
        self.se_block = _SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = _Conv2d_IBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = _Conv2d_IBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = _Conv2d_IBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x] 


class _HRBranch(pytorch_lightning.LightningModule):
    """ High Resolution Branch of _MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(_HRBranch, self).__init__()

        self.tohr_enc2x = _Conv2d_IBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x = _Conv2d_IBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

        self.tohr_enc4x = _Conv2d_IBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x = _Conv2d_IBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

        self.conv_hr4x = nn.Sequential(
            _Conv2d_IBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr2x = nn.Sequential(
            _Conv2d_IBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
        )

        self.conv_hr = nn.Sequential(
            _Conv2d_IBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)

        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


class _FusionBranch(pytorch_lightning.LightningModule):
    """ Fusion Branch of _MODNet
    """

    def __init__(self, hr_channels, enc_channels):
        super(_FusionBranch, self).__init__()
        self.conv_lr4x = _Conv2d_IBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        
        self.conv_f2x = _Conv2d_IBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f = nn.Sequential(
            _Conv2d_IBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            _Conv2d_IBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte

class _MODNet(pytorch_lightning.LightningModule):
    """ Architecture of _MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True):
        super(_MODNet, self).__init__()

        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels)

        self.lr_branch = _LRBranch(self.backbone)
        self.hr_branch = _HRBranch(self.hr_channels, self.backbone.enc_channels)
        self.f_branch = _FusionBranch(self.hr_channels, self.backbone.enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img, inference):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

class _GaussianBlurLayer(pytorch_lightning.LightningModule):
    """ Add Gaussian Blur to a 4D tensors
    This layer takes a 4D tensor of {N, C, H, W} as input.
    The Gaussian blur will be performed in given channel number (C) splitly.
    """

    def __init__(self, channels, kernel_size):
        """ 
        Arguments:
            channels (int): Channel for input tensor
            kernel_size (int): Size of the kernel used in blurring
        """

        super(_GaussianBlurLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 != 0

        self.op = nn.Sequential(
            nn.ReflectionPad2d(math.floor(self.kernel_size / 2)), 
            nn.Conv2d(channels, channels, self.kernel_size, 
                      stride=1, padding=0, bias=None, groups=channels)
        )

        self._init_kernel()

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): input 4D tensor
        Returns:
            torch.Tensor: Blurred version of the input 
        """

        if not len(list(x.shape)) == 4:
            print('\'GaussianBlurLayer\' requires a 4D tensor as input\n')
            exit()
        elif not x.shape[1] == self.channels:
            print('In \'GaussianBlurLayer\', the required channel ({0}) is'
                  'not the same as input ({1})\n'.format(self.channels, x.shape[1]))
            exit()
            
        return F.sigmoid(self.op(x))
    
    def _init_kernel(self):
        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8

        n = np.zeros((self.kernel_size, self.kernel_size))
        i = math.floor(self.kernel_size / 2)
        n[i, i] = 1
        kernel = scipy.ndimage.gaussian_filter(n, sigma)

        for name, param in self.named_parameters():
            param.data.copy_(torch.from_numpy(kernel))
