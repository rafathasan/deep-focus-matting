from typing import List
import pytorch_lightning
import torch
import math
import scipy
import torchvision
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from networks.deeplabv3_focus.deeplabv3plus import DeepLabV3Plus
from networks.gfm_util import *
from networks.Wrapper import LightningWrapper

class DFM(LightningWrapper):
    def __init__(
        self,
        settings = None
    ):
        super().__init__(settings)
        # self.net = torch.nn.DataParallel(DeepLabV3Plus(3))
        self.net = DeepLabV3Plus(3)
        # self.blurer = _GaussianBlurLayer(1, 3)

    def forward(self, x):
        return self.net(x)

    def calc_loss(self, image, mask, trimap, fg, bg, predict_global, predict_local, predict_fusion):
        gt_copy = mask.clone()
        gt_copy[gt_copy==0] = 0
        gt_copy[gt_copy==255] = 2
        gt_copy[gt_copy>2] = 1
        gt_copy = gt_copy.long()
        gt_copy = gt_copy[:,0,:,:]
        criterion = nn.CrossEntropyLoss()
        entropy_loss = criterion(predict_global, gt_copy)

        # diffrence square root approximation / unknown region = 1
        weighted = torch.zeros(trimap.shape)
        weighted[trimap == 128] = 1.
        alpha_f = mask / 255.
        alpha_f = alpha_f
        diff = predict_local - alpha_f
        diff = diff * weighted
        alpha_loss = torch.sqrt(diff ** 2 + 1e-6) # 1e-12
        alpha_loss_weighted = alpha_loss.sum() / (weighted.sum() + 1.)

        # laplacian loss in unknown region
        weighted = torch.zeros(trimap.shape)
        weighted[trimap == 128] = 1.
        alpha_f = mask / 255.
        alpha_f = alpha_f
        alpha_f = alpha_f.clone()*weighted
        predict = predict_local.clone()*weighted
        gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
        pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
        pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
        laplacian_loss_weighted = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        # diffrence square root approximation / mask = 1
        weighted = torch.ones(mask.shape)
        alpha_f = mask / 255.
        alpha_f = alpha_f
        diff = predict_fusion - alpha_f
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
        alpha_loss = alpha_loss.sum()/(weighted.sum())

        alpha_f = mask / 255.
        alpha_f = alpha_f
        gauss_kernel = build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=True)
        pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
        pyr_predict = laplacian_pyramid(predict_fusion, gauss_kernel, 5)
        laplacian_loss = sum(fnn.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))

        weighted = torch.ones(mask.shape).cuda()
        predict_3 = torch.cat((predict_fusion, predict_fusion, predict_fusion), 1)
        comp = predict_3 * fg + (1. - predict_3) * bg
        comp_loss = torch.sqrt((comp - image) ** 2 + 1e-12) / 255.
        comp_loss = comp_loss.sum()/(weighted.sum())

        return 0.25*entropy_loss+0.25*(alpha_loss_weighted+laplacian_loss_weighted)+0.25*(laplacian_loss_weighted*alpha_loss)+0.25*comp_loss


        
    def training_step(self, batch, batch_idx):
        image, mask, trimap, fg, bg = batch

        predict_global, predict_local, predict_fusion = self(image)

        # predict_global = self.blurer(predict_global)

        loss_global = get_crossentropy_loss(3, trimap, predict_global)

        # loss_global = F.mse_loss(predict_global[:,1,:,:], target=mask)

        loss_local = get_alpha_loss(predict_local, mask, trimap) # diffrence square root approximation / unknown region = 1
        + get_laplacian_loss(predict_local, mask, trimap) # laplacian loss in unknown region

        loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) # diffrence square root approximation / mask = 1
        + get_laplacian_loss_whole_img(predict_fusion, mask)
        loss_fusion_comp = get_composition_loss_whole_img(image, mask, fg, bg, predict_fusion)

        loss = .25*loss_global+.25*loss_local+.25*loss_fusion_alpha#+.25*loss_fusion_comp

        if batch_idx == 0:
            # self.log_image(title="training_images", predict=predict_fusion, mask=mask)
            self.logger.experiment.add_image("training_images", torchvision.utils.make_grid(torch.cat((predict_fusion, mask/255),2)), self.current_epoch)
            self.logger.experiment.add_image("training_global", torchvision.utils.make_grid(predict_global), self.current_epoch)
            # self.log_image(title="training_local", predict=predict_local, mask=mask)
            self.logger.experiment.add_image("training_local", torchvision.utils.make_grid(predict_local), self.current_epoch)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=predict_fusion, mask=mask),
            "l2loss": self.l2loss(predict=predict_fusion, mask=mask),
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg = batch

            predict_global, predict_local, predict_fusion = self(image)

            loss_global = get_crossentropy_loss(3, trimap, predict_global)
            loss_local = get_alpha_loss(predict_local, mask, trimap) + get_laplacian_loss(predict_local, mask, trimap)

            loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
            loss_fusion_comp = get_composition_loss_whole_img(image, mask, fg, bg, predict_fusion)
            loss = 0.25*loss_global+0.25*loss_local +0.25*loss_fusion_alpha+0.25*loss_fusion_comp

            if batch_idx == 0:
                self.log_image(title="validation_images", predict=predict_fusion, mask=mask)
                self.logger.experiment.add_image("validation_global", torchvision.utils.make_grid(predict_global), self.current_epoch)
                self.log_image(title="validation_local", predict=predict_local, mask=mask)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=predict_fusion, mask=mask),
            "l2loss": self.l2loss(predict=predict_fusion, mask=mask),
        }

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            torch.cuda.empty_cache()
            image, mask, trimap, fg, bg, h, w = batch

            self.starter.record()
            predict_global, predict_local, predict_fusion = self(image)
            self.ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            inference_time = self.starter.elapsed_time(self.ender)
            resize = torchvision.transforms.Resize((h,w))

            image = resize(image)
            predict_fusion = resize(predict_fusion)
            mask = resize(mask)

            return self.l1loss(predict=predict_fusion, mask=mask),  self.l2loss(predict=predict_fusion, mask=mask), inference_time*1e-3, image, predict_fusion, mask

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
