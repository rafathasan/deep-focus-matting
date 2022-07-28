import pytorch_lightning
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from networks.deeplabv3_focus.deeplabv3plus import DeepLabV3Plus
from networks.gfm_util import *

def create_optimizers(self):
    optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.lr_sc_factor,
            patience=self.lr_sc_patience,
            verbose=True,
        )
    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "monitor": "val_loss",
    }


class DFM(pytorch_lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        lr_sc_factor: float,
        lr_sc_patience: float,
    ):
        super(DFM, self).__init__()
        self.net = DeepLabV3Plus(3)
        self.learning_rate = learning_rate
        self.lr_sc_factor = lr_sc_factor
        self.lr_sc_patience = lr_sc_patience

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask, trimap, fg, bg = batch
        image = image.float()
        mask = mask.float()
        trimap = trimap.float()
        fg = fg.float()
        bg = bg.float()

        predict_global, predict_local, predict_fusion = self(image)
        # loss = F.mse_loss(predict, mask)

        loss_global = get_crossentropy_loss(3, trimap, predict_global)
        loss_local = get_alpha_loss(predict_local, mask, trimap) + get_laplacian_loss(predict_local, mask, trimap)

        loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
        loss_fusion_comp = get_composition_loss_whole_img(image, mask, fg, bg, predict_fusion)
        loss = 0.25*loss_global+0.25*loss_local+0.25*loss_fusion_alpha+0.25*loss_fusion_comp

        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        self.log("train_loss", train_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg = batch
            image = image.float()
            mask = mask.float()
            trimap = trimap.float()
            fg = fg.float()
            bg = bg.float()

            predict_global, predict_local, predict_fusion = self(image)
            # loss = F.mse_loss(predict, mask)

            loss_global = get_crossentropy_loss(3, trimap, predict_global)
            loss_local = get_alpha_loss(predict_local, mask, trimap) + get_laplacian_loss(predict_local, mask, trimap)

            loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
            loss_fusion_comp = get_composition_loss_whole_img(image, mask, fg, bg, predict_fusion)
            loss = 0.25*loss_global+0.25*loss_local+0.25*loss_fusion_alpha+0.25*loss_fusion_comp

        return {
            "vloss": loss,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.Tensor([output["vloss"] for output in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return create_optimizers(self)