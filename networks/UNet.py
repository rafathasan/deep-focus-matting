import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.Wrapper import LightningWrapper
import torchvision

class UNet(LightningWrapper):
    def __init__(
        self,
        settings = None,
    ):
        super().__init__(settings)
        self.net = _UNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        image, mask, trimap, fg, bg = batch
        predict = torch.sigmoid(self(image))
        loss = F.mse_loss(predict, mask/255)

        if batch_idx == 0:
            self.log_image(title="training_images", predict=predict, mask=torch.relu(mask))

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=predict, mask=mask),
            "l2loss": self.l2loss(predict=predict, mask=mask),
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg = batch
            predict = torch.sigmoid(self(image))
            loss = F.mse_loss(predict, mask) / (mask.shape[0]*mask.shape[1]*mask.shape[2]*mask.shape[3])

            if batch_idx == 0:
                self.log_image(title="validation_images", predict=predict, mask=mask)

        return {
            "loss": loss,
            "l1loss": self.l1loss(predict=predict, mask=mask),
            "l2loss": self.l2loss(predict=predict, mask=mask),
        }

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            image, mask, trimap, fg, bg, h, w = batch

            torch.cuda.empty_cache()

            self.starter.record()
            predict = torch.sigmoid(self(image))
            self.ender.record()
            # wait for gpu sync
            torch.cuda.synchronize()
            inference_time = self.starter.elapsed_time(self.ender)
            resize = torchvision.transforms.Resize((h,w))

            image = resize(image)
            predict = resize(predict)
            mask = resize(mask)

            return self.l1loss(predict=predict, mask=mask),  self.l2loss(predict=predict, mask=mask), inference_time*1e-3, image, predict, mask

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