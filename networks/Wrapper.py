import pytorch_lightning
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from abc import ABC, abstractmethod

def create_optimizers(self, settings):
    # optimizer = torch.optim.Adam(
    #         self.parameters(),
    #         lr=settings["learning_rate"],
    #         # weight_decay=1e-6
    #     )
    optimizer = torch.optim.SGD(
            self.parameters(),
            lr=settings["learning_rate"],
            momentum=0.9,
        )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=.1,
            patience=2,
            verbose=True,
        )
    # lr_scheduler = optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=10,
    #         gamma=.1,
    #         verbose=False,
    #     )
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=5,
    #         eta_min=1e-5,
    #         verbose=False,
    #     )

    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "monitor": settings["monitor"],
    }

class LightningWrapper(pytorch_lightning.LightningModule, ABC):
    def __init__(self, settings):
        super(LightningWrapper, self).__init__()
        self.settings = settings
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    def l1loss(self, predict, mask):
        return F.l1_loss(predict, mask/255)

    def l2loss(self, predict, mask):
        return F.mse_loss(predict, mask/255)

    def log_image(self, title, predict, mask):
        grid = torchvision.utils.make_grid(torch.cat((predict, mask/255), 2), padding=0)

        self.logger.experiment.add_image(title, grid, self.current_epoch)

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass
    @abstractmethod
    def training_epoch_end(self, training_outputs):
        pass
    def training_epoch_end(self, training_outputs):
        training_l1loss = torch.Tensor([output["l1loss"] for output in training_outputs]).mean()
        training_l2loss = torch.Tensor([output["l2loss"] for output in training_outputs]).mean()
        training_loss = torch.Tensor([output["loss"] for output in training_outputs]).mean()
        self.log("training_l1loss", training_l1loss, prog_bar=False)
        self.log("training_l2loss", training_l2loss, prog_bar=False)
        self.log("training_loss", training_loss, prog_bar=False)

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass
    def validation_epoch_end(self, validation_outputs):
        validation_l1loss = torch.Tensor([output["l1loss"] for output in validation_outputs]).mean()
        validation_l2loss = torch.Tensor([output["l2loss"] for output in validation_outputs]).mean()
        validation_loss = torch.Tensor([output["loss"] for output in validation_outputs]).mean()
        self.log("validation_l1loss", validation_l1loss, prog_bar=False)
        self.log("validation_l2loss", validation_l2loss, prog_bar=False)
        self.log("validation_loss", validation_loss, prog_bar=False)
        
    @abstractmethod
    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return create_optimizers(self, self.settings)