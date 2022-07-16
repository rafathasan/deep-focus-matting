import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class UNet(pytorch_lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        lr_sc_factor: float,
        lr_sc_patience: float,
    ):
        super(UNet, self).__init__()
        self.net = _UNet()
        self.learning_rate = learning_rate
        self.lr_sc_factor = lr_sc_factor
        self.lr_sc_patience = lr_sc_patience

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask, _ = batch
        image = image/255
        mask = mask/255
        predict = torch.sigmoid(self(image))
        loss = F.mse_loss(predict, mask)

        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        self.log("train_loss", train_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, _ = batch
            image = image/255
            mask = mask/255
            predict = torch.sigmoid(self(image))
            loss = F.mse_loss(predict, mask)

        return {
            "vloss": loss,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.Tensor([output["vloss"] for output in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return create_optimizers(self)

class _DoubleConv(pytorch_lightning.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class _UNet(pytorch_lightning.LightningModule):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(_DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(_DoubleConv(feature*2, feature))

        self.bottleneck = _DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)