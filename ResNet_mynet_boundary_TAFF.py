import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models as models
from thop import profile
from Res2Net_U.model import FCM
from Res2Net_U.model import TAFF
from Res2Net_U.model import Conv2d_cd


class ResNet_mynet_boundary_TAFF(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=1):
        super(ResNet_mynet_boundary_TAFF, self).__init__()
        # ---- Front Network----#
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 8, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.Decoder1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True))

        self.Decoder2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # ---- Res2Net50 Backbone ----
        self.resnet = models.resnet50(pretrained=True)

        self.conv_boun_f = nn.Sequential(
            Conv2d_cd.Conv2d_cd(3,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.conv_boun_c1 = nn.Sequential(
            Conv2d_cd.Conv2d_cd(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.boun_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     )
        self.conv_boun_c2 = nn.Sequential(
            Conv2d_cd.Conv2d_cd(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.boun_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.con_boun_out = nn.Conv2d(32, 1, 1)
        self.pool_b = nn.MaxPool2d(2, 2)
        self.conv1_bon1x1 = nn.Conv2d(32, 256, 1)
        self.conv2_bon1x1 = nn.Conv2d(64, 512, 1)
        self.conv3_bon1x1 = nn.Conv2d(128, 1024, 1)


        # ---- Receptive Field Block like module ----
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 1, 1)
                      )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 256, 1, 1)
                      )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 256, 1, 1)
                      )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 1, 1)
                      )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 512, 1, 1)
                      )

        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(2048, 1024, 1, 1)
                      )

        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)

        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU(inplace=True)

        self.espcn1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(64,32,3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        self.espcn2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

        self.espcn3 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 2, 3, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 1),
        )

        # self.taff1 = TAFF.AFF(256)
        # self.taff2 = TAFF.AFF(512)
        # self.taff3 = TAFF.AFF(1024)


    def forward(self, x):
        x_Front1 = self.conv1(x)
        x_Front2 = self.conv2(x_Front1)
        x_Front3 = self.Decoder1(x_Front2)
        x = torch.cat([x_Front1,x_Front3],dim=1)
        x = self.conv3(x)
        x = self.Decoder2(x)
        x = self.conv4(x)

        x_boun_f= self.conv_boun_f(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x) # 64*64*64
        x_max = self.resnet.maxpool(x)


        x_cat = torch.cat([x_boun_f, x_max],dim=1)
        x_b_1 = self.conv_boun_c1(x_cat)
        x_boun_up = self.boun_up(x_b_1)
        x_b_2 = self.conv_boun_c2(x_boun_up)
        x_boun_up2 = self.boun_up2(x_b_2)

        x_b_out = self.con_boun_out(x_boun_up2)

        x_add1 = self.pool_b(self.conv1_bon1x1(x_boun_up2))
        x_add2 = self.pool_b(self.conv2_bon1x1(x_boun_up))
        x_add3 = self.pool_b(self.conv3_bon1x1(x_cat))

        # ---- low-level features ----
        x1 = self.resnet.layer1(x) # 256*64*64
        x2 = self.resnet.layer2(x1) # 512*32*32

        x3 = self.resnet.layer3(x2) # 1024*16*16
        x4 = self.resnet.layer4(x3) # 2048*8*8

        x2_up = self.up1(x2)
        x3_up = self.up2(x3)
        x4_up = self.up3(x4)

        x3_up_2 = self.up4(x3)
        x4_up_2 = self.up5(x4)

        x4_up_3 = self.up6(x4)

        all1 = self.relu1(self.bn1(x1 + x2_up + x3_up + x4_up + x_add1))
        all2 = self.relu2(self.bn2(x2 + x3_up_2 + x4_up_2 + x_add2))
        all3 = self.relu3(self.bn3(x3 + x4_up_3 + x_add3))

        out1 = self.espcn1(all1)
        out2 = self.espcn2(all2)
        out3 = self.espcn3(all3)


        out = out1 + out2 + out3

        return out, x_b_out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ras = ResNet_mynet_boundary_TAFF(1).to(device)
    print(summary(ras, (1,128,128)))

    input = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(ras, inputs=(input,))
    print(f"FLOPs: {flops / 10 ** 9} G")
