import pytorch_lightning
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
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


class GFM(pytorch_lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        lr_sc_factor: float,
        lr_sc_patience: float,
    ):
        super(GFM, self).__init__()
        self.net = _GFM(backbone='r34', rosta='TT')
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

        mse = F.mse_loss(predict_fusion, mask)

        return {
            "loss": loss,
            "mse": mse,
        }

    def training_epoch_end(self, outputs):
        train_loss = torch.Tensor([output["loss"] for output in outputs]).mean()
        mse_loss = torch.Tensor([output["mse"] for output in outputs]).mean()
        self.log("train_loss", train_loss, prog_bar=True)
        self.log("mse_loss", mse_loss, prog_bar=True)

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

            mse = F.mse_loss(predict_fusion, mask)

        return {
            "vloss": loss,
            "vmse": mse,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.Tensor([output["vloss"] for output in outputs]).mean()
        eval_mse_loss = torch.Tensor([output["vmse"] for output in outputs]).mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("eval_mse_loss", eval_mse_loss, prog_bar=True)

    def configure_optimizers(self):
        return create_optimizers(self)

def collaborative_matting(rosta, glance_sigmoid, focus_sigmoid):
	if rosta =='TT':
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
	elif rosta == 'BT':
		values, index = torch.max(glance_sigmoid,1)
		index = index[:,None,:,:].float()
		fusion_sigmoid = index - focus_sigmoid
		fusion_sigmoid[fusion_sigmoid<0]=0
	else:
		values, index = torch.max(glance_sigmoid,1)
		index = index[:,None,:,:].float()
		fusion_sigmoid = index + focus_sigmoid
		fusion_sigmoid[fusion_sigmoid>1]=1
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

class _GFM(pytorch_lightning.LightningModule):

    def __init__(self, backbone, rosta):
        super().__init__()

        self.backbone = backbone
        self.rosta = rosta
        if self.rosta=='TT':
            self.gd_channel = 3
        else:
            self.gd_channel = 2

        if self.backbone=='r34_2b':
            ##################################
            ### Backbone - Resnet34 + 2 blocks
            ##################################
            self.resnet = models.resnet34(pretrained=True)
            self.encoder0 = nn.Sequential(
                nn.Conv2d(3,64,3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
            self.encoder1 = self.resnet.layer1
            self.encoder2 = self.resnet.layer2
            self.encoder3 = self.resnet.layer3
            self.encoder4 = self.resnet.layer4
            self.encoder5 = nn.Sequential(
                nn.MaxPool2d(2, 2, ceil_mode=True), 
                BasicBlock(512,512),
                BasicBlock(512,512),
                BasicBlock(512,512))
            self.encoder6 = nn.Sequential(
                nn.MaxPool2d(2, 2, ceil_mode=True),
                BasicBlock(512,512),
                BasicBlock(512,512),
                BasicBlock(512,512))

            self.psp_module = PSPModule(512, 512, (1, 3, 5))
            self.psp6 = conv_up_psp(512, 512, 2)
            self.psp5 = conv_up_psp(512, 512, 4)
            self.psp4 = conv_up_psp(512, 256, 8)
            self.psp3 = conv_up_psp(512, 128, 16)
            self.psp2 = conv_up_psp(512, 64, 32)
            self.psp1 = conv_up_psp(512, 64, 32)
            self.decoder6_g = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder5_g = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder4_g = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_g = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_g = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_g = build_decoder(128, 64, 64, 64, True, False)
            
            self.bridge_block = build_bb(512, 512, 512)
            self.decoder6_f = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder5_f = build_decoder(1024, 512, 512, 512, True, True)
            self.decoder4_f = build_decoder(1024, 512, 512, 256, True, True)
            self.decoder3_f = build_decoder(512, 256, 256, 128, True, True)
            self.decoder2_f = build_decoder(256, 128, 128, 64, True, True)
            self.decoder1_f = build_decoder(128, 64, 64, 64, True, False)

            if self.rosta == 'RIM':
                self.decoder0_g_tt = nn.Sequential(nn.Conv2d(64,3,3,padding=1))
                self.decoder0_g_ft = nn.Sequential(nn.Conv2d(64,2,3,padding=1))
                self.decoder0_g_bt = nn.Sequential(nn.Conv2d(64,2,3,padding=1))
                self.decoder0_f_tt = nn.Sequential(nn.Conv2d(64,1,3,padding=1))
                self.decoder0_f_ft = nn.Sequential(nn.Conv2d(64,1,3,padding=1))
                self.decoder0_f_bt = nn.Sequential(nn.Conv2d(64,1,3,padding=1))
            else:
                self.decoder0_g = nn.Sequential(nn.Conv2d(64,self.gd_channel,3,padding=1))
                self.decoder0_f = nn.Sequential(nn.Conv2d(64,1,3,padding=1))


        if self.backbone=='r34':
            ##########################
            ### Backbone - Resnet34
            ##########################
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
            
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)


        elif self.backbone=='r101':
            ##########################
            ### Backbone - Resnet101
            ##########################
            self.resnet = models.resnet101(pretrained=True)
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
            self.psp_module = PSPModule(2048, 2048, (1, 3, 5))
            self.bridge_block = build_bb(2048, 2048, 2048)
            self.psp4 = conv_up_psp(2048, 1024, 2)
            self.psp3 = conv_up_psp(2048, 512, 4)
            self.psp2 = conv_up_psp(2048, 256, 8)
            self.psp1 = conv_up_psp(2048, 64, 16)

            self.decoder4_g = build_decoder(4096, 2048, 1024, 1024, True, True)
            self.decoder3_g = build_decoder(2048, 1024, 512, 512, True, True)
            self.decoder2_g = build_decoder(1024, 512, 256, 256, True, True)
            self.decoder1_g = build_decoder(512, 256, 128, 64, True, True)
            
            self.decoder4_f = build_decoder(4096, 2048, 1024, 1024, True, True)
            self.decoder3_f = build_decoder(2048, 1024, 512, 512, True, True)
            self.decoder2_f = build_decoder(1024, 512, 256, 256, True, True)
            self.decoder1_f = build_decoder(512, 256, 128, 64, True, True)
            
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)


        elif self.backbone=='d121':
            #############################
            ### Encoder part - DESNET121
            #############################
            self.densenet = models.densenet121(pretrained=True)
            self.encoder0 = nn.Sequential(
                self.densenet.features.conv0,
                self.densenet.features.norm0,
                self.densenet.features.relu0)
            self.encoder1 = nn.Sequential(
                self.densenet.features.denseblock1,
                self.densenet.features.transition1)
            self.encoder2 = nn.Sequential(
                self.densenet.features.denseblock2,
                self.densenet.features.transition2)
            self.encoder3 = nn.Sequential(
                self.densenet.features.denseblock3,
                self.densenet.features.transition3)
            self.encoder4 = nn.Sequential(
                self.densenet.features.denseblock4,
                nn.Conv2d(1024,512,3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, ceil_mode=True))
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
            self.decoder3_f = build_decoder(768, 256, 256, 128, True, True)
            self.decoder2_f = build_decoder(384, 128, 128, 64, True, True)
            self.decoder1_f = build_decoder(192, 64, 64, 64, True, True)
            
            if self.rosta == 'RIM':
                self.decoder0_g_tt = build_decoder(128, 64, 64, 3, False, True)
                self.decoder0_g_ft = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_g_bt = build_decoder(128, 64, 64, 2, False, True)
                self.decoder0_f_tt = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_ft = build_decoder(128, 64, 64, 1, False, True)
                self.decoder0_f_bt = build_decoder(128, 64, 64, 1, False, True)
            else:
                self.decoder0_g = build_decoder(128, 64, 64, self.gd_channel, False, True)
                self.decoder0_f = build_decoder(128, 64, 64, 1, False, True)

        if self.rosta=='RIM':
            self.rim = nn.Sequential(
                nn.Conv2d(3,16,1),
                SELayer(16),
                nn.Conv2d(16,1,1))
        
    def forward(self, input):

        glance_sigmoid = torch.zeros(input.shape)
        focus_sigmoid =  torch.zeros(input.shape)
        fusion_sigmoid =  torch.zeros(input.shape)

        #################
        ### Encoder part 
        #################
        e0 = self.encoder0(input)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        ##########################
        ### Decoder part - GLANCE
        ##########################
        if self.backbone=='r34_2b':
            e5 = self.encoder5(e4)
            e6 = self.encoder6(e5)
            psp = self.psp_module(e6) 
            d6_g = self.decoder6_g(torch.cat((psp, e6),1))
            d5_g = self.decoder5_g(torch.cat((self.psp6(psp),d6_g),1))
            d4_g = self.decoder4_g(torch.cat((self.psp5(psp),d5_g),1))
        else:
            psp = self.psp_module(e4)
            d4_g = self.decoder4_g(torch.cat((psp,e4),1))
        
        d3_g = self.decoder3_g(torch.cat((self.psp4(psp),d4_g),1))
        d2_g = self.decoder2_g(torch.cat((self.psp3(psp),d3_g),1))
        d1_g = self.decoder1_g(torch.cat((self.psp2(psp),d2_g),1))

        if self.backbone=='r34_2b':
            if self.rosta=='RIM':
                d0_g_tt = self.decoder0_g_tt(d1_g)
                d0_g_ft = self.decoder0_g_ft(d1_g)
                d0_g_bt = self.decoder0_g_bt(d1_g)
            else:
                d0_g = self.decoder0_g(d1_g)
        else:
            if self.rosta=='RIM':
                d0_g_tt = self.decoder0_g_tt(torch.cat((self.psp1(psp),d1_g),1))
                d0_g_ft = self.decoder0_g_ft(torch.cat((self.psp1(psp),d1_g),1))
                d0_g_bt = self.decoder0_g_bt(torch.cat((self.psp1(psp),d1_g),1))
            else:
                d0_g = self.decoder0_g(torch.cat((self.psp1(psp),d1_g),1))
        
        if self.rosta=='RIM':
            glance_sigmoid_tt = F.sigmoid(d0_g_tt)
            glance_sigmoid_ft = F.sigmoid(d0_g_ft)
            glance_sigmoid_bt = F.sigmoid(d0_g_bt)
        else:
            glance_sigmoid = F.sigmoid(d0_g)


        ##########################
        ### Decoder part - FOCUS
        ##########################
        if self.backbone == 'r34_2b':
            bb = self.bridge_block(e6)
            d6_f = self.decoder6_f(torch.cat((bb, e6),1))
            d5_f = self.decoder5_f(torch.cat((d6_f, e5),1))
            d4_f = self.decoder4_f(torch.cat((d5_f, e4),1))
        else:
            bb = self.bridge_block(e4)
            d4_f = self.decoder4_f(torch.cat((bb, e4),1))
        d3_f = self.decoder3_f(torch.cat((d4_f, e3),1))    
        d2_f = self.decoder2_f(torch.cat((d3_f, e2),1))
        d1_f = self.decoder1_f(torch.cat((d2_f, e1),1))

        if self.backbone=='r34_2b':
            if self.rosta=='RIM':
                d0_f_tt = self.decoder0_f_tt(d1_f)
                d0_f_ft = self.decoder0_f_ft(d1_f)
                d0_f_bt = self.decoder0_f_bt(d1_f)
            else:
                d0_f = self.decoder0_f(d1_f)
        else:
            if self.rosta=='RIM':
                d0_f_tt = self.decoder0_f_tt(torch.cat((d1_f, e0),1))
                d0_f_ft = self.decoder0_f_ft(torch.cat((d1_f, e0),1))
                d0_f_bt = self.decoder0_f_bt(torch.cat((d1_f, e0),1))
            else:
                d0_f = self.decoder0_f(torch.cat((d1_f, e0),1))
        
        if self.rosta=='RIM':
            focus_sigmoid_tt = F.sigmoid(d0_f_tt)
            focus_sigmoid_ft = F.sigmoid(d0_f_ft)
            focus_sigmoid_bt = F.sigmoid(d0_f_bt)
        else:
            focus_sigmoid = F.sigmoid(d0_f)
        
        ##########################
        ### Collaborative Matting
        ##########################
        if self.rosta=='RIM':
            fusion_sigmoid_tt = collaborative_matting('TT', glance_sigmoid_tt, focus_sigmoid_tt)
            fusion_sigmoid_ft = collaborative_matting('FT', glance_sigmoid_ft, focus_sigmoid_ft)
            fusion_sigmoid_bt = collaborative_matting('BT', glance_sigmoid_bt, focus_sigmoid_bt)
            fusion_sigmoid = torch.cat((fusion_sigmoid_tt,fusion_sigmoid_ft,fusion_sigmoid_bt),1)
            fusion_sigmoid = self.rim(fusion_sigmoid)
            return [[glance_sigmoid_tt, focus_sigmoid_tt, fusion_sigmoid_tt],[glance_sigmoid_ft, focus_sigmoid_ft, fusion_sigmoid_ft],[glance_sigmoid_bt, focus_sigmoid_bt, fusion_sigmoid_bt], fusion_sigmoid]
        else:
            fusion_sigmoid = collaborative_matting(self.rosta, glance_sigmoid, focus_sigmoid)
            return glance_sigmoid, focus_sigmoid, fusion_sigmoid