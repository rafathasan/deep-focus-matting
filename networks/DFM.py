from typing import List
import pytorch_lightning
import torch
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
        self.net = DeepLabV3Plus(3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask, trimap, fg, bg = batch

        predict_global, predict_local, predict_fusion = self(image)

        loss_global = get_crossentropy_loss(3, trimap, predict_global)
        loss_local = get_alpha_loss(predict_local, mask, trimap) + get_laplacian_loss(predict_local, mask, trimap)

        loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
        loss_fusion_comp = get_composition_loss_whole_img(image, mask, fg, bg, predict_fusion)
        loss = 0.25*loss_global+0.25*loss_local+0.25*loss_fusion_alpha+0.25*loss_fusion_comp

        self.log_image(title="training_images", predict=predict_fusion, mask=mask)

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
            loss = 0.25*loss_global+0.25*loss_local+0.25*loss_fusion_alpha+0.25*loss_fusion_comp

            self.log_image(title="validation_images", predict=predict_fusion, mask=mask)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=predict_fusion, mask=mask),
            "l2loss": self.l2loss(predict=predict_fusion, mask=mask),
        }

    def predict_step(self, batch, batch_idx):
        pass