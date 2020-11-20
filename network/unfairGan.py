from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.utils.data


act = nn.LeakyReLU(0.1, inplace=True)

###  Generator
class Generator(nn.Module):

    def __init__(self, inX_chs=3, inRM_chs=1, inED_chs=3, out_chs=3, nfeats=32, num_main_blk=1,
                 nDlayer=4, grRate=32, act_type='DAF', #['DAF', 'XU', 'ReLU']
                 mainblock_type='U_D', #['U_D', 'U']
                 ):
        super(Generator, self).__init__()
        # rm=Rain Map, ED=Edge
        self.num_blk = num_main_blk
        if inRM_chs >0 and inED_chs>0:
            auam_nfeats = int(nfeats / 2)
            condRM = int(auam_nfeats * 0.125)
            condED = auam_nfeats - condRM
        elif inRM_chs > 0:
            auam_nfeats = int(nfeats / 2)
            condRM = auam_nfeats
            condED = 0
        elif inED_chs > 0:
            auam_nfeats = int(nfeats / 2)
            condRM = 0
            condED = auam_nfeats
        else:
            auam_nfeats = 0
            condRM = 0
            condED = 0

        if auam_nfeats >0:
            self.auam = AuAM(auam_nfeats, act_type=act_type)
        else:
            self.auam = None

        self.aam = AAM(inX_chs=inX_chs, inRM_chs=inRM_chs, inED_chs=inED_chs, nfeats=nfeats,
                       condRM=condRM, condED=condED, act_type=act_type)

        modules = []
        for i in range(self.num_blk):
            modules.append(mainBlock(nfeat=nfeats, nDlayer=nDlayer, grRate=grRate, auam=self.auam, block_type=mainblock_type))
        self.main_blk = nn.Sequential(*modules)

        self.out = nn.Sequential(
                act,
                nn.Conv2d(nfeats, out_chs, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, x, rm=None, ed=None):
        if rm is not None or ed is not None:
            fea_x, fea_sub = self.aam(x, rm, ed)
            for main in self.main_blk:
                fea_x = main(fea_x, fea_sub)
            out = self.out(self.auam((fea_x, fea_sub))) + x
        else:
            fea_x = self.aam(x)
            for main in self.main_blk:
                fea_x = main(fea_x)
            out = self.out(fea_x) + x

        return out

class Discriminator(nn.Module):
    def __init__(self, inX_chs=3, nfeats=32, inRM_chs=1, inED_chs=3, act_type='DAF',):
        super(Discriminator, self).__init__()
        # rm=Rain Map, ED=Edge
        if inRM_chs > 0 and inED_chs > 0:
            condfeat = int(nfeats / 2)
            condRM = int(condfeat * 0.125)
            condED = condfeat - condRM
        elif inRM_chs > 0:
            condfeat = int(nfeats / 2)
            condRM = condfeat
            condED = 0
        elif inED_chs > 0:
            condfeat = int(nfeats / 2)
            condRM = 0
            condED = condfeat
        else:
            condfeat = 0
            condRM = 0
            condED = 0

        if condfeat > 0:
            self.condblock = AuAM(condfeat, act_type=act_type)
        else:
            self.condblock = None

        self.in_blk = AAM(inX_chs=inX_chs, inRM_chs=inRM_chs, inED_chs=inED_chs, nfeats=nfeats,
                          condRM=condRM, condED=condED, act_type=act_type)

        self.main = nn.Sequential(
            nn.Conv2d(nfeats, nfeats, kernel_size=5, stride=2, padding=2),
            act,
            nn.Conv2d(nfeats, nfeats * 2, kernel_size=5, padding=2),
            act,
            nn.Conv2d(nfeats * 2, nfeats * 2, kernel_size=5, stride=2, padding=2),
            act,
            nn.Conv2d(nfeats * 2, nfeats * 4, kernel_size=5, padding=2),
            act,
            nn.Conv2d(nfeats * 4, nfeats * 4, kernel_size=5, stride=2, padding=2),
            act,
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nfeats * 4, nfeats * 8, kernel_size=1),
            act,
            nn.Conv2d(nfeats * 8, 1, kernel_size=1),
        )

        # self.fc = nn.Linear(4 * 4 * 1024, 1)

    def forward(self, x, rm=None, ed=None):
        if rm is not None or ed is not None:
            fea_x, fea_sub = self.in_blk(x, rm, ed)
            y = self.main(self.condblock((fea_x, fea_sub)))
        else:
            fea_x = self.inx(x)
            y = self.main(fea_x)
        si = y.view(y.size()[0], -1)
        return si


### sub class
class AAM(nn.Module):

    def __init__(self, inX_chs=3, nfeats=32, inRM_chs=1, inED_chs=3,
                 condRM=4, condED=12, act_type='DAF',):
        super(AAM, self).__init__()
        if act_type =='DAF':
            act_inX = DAF(nfeats, nfeats, dilations=[1, 3, 6, 12])
            act_inRM = DAF(condRM, condRM, dilations=[4])
            act_inED = DAF(condED, condED, dilations=[4])
        elif act_type == 'XU':
            act_inX = Modulecell(nfeats, nfeats)
            act_inRM = Modulecell(condRM, condRM)
            act_inED = Modulecell(condED, condED)
        elif act_type == 'ReLU':
            act_inX = nn.ReLU()
            act_inRM = nn.ReLU()
            act_inED = nn.ReLU()

        self.inx = nn.Sequential(
            nn.Conv2d(inX_chs, nfeats, kernel_size=3, stride=1, padding=1),
            act_inX,
        )
        if inRM_chs > 0:
            self.inRM = nn.Sequential(
                nn.Conv2d(inRM_chs, condRM, kernel_size=3, stride=1, padding=1),
                act_inRM,
            )
        if inED_chs > 0:
            self.inED = nn.Sequential(
                nn.Conv2d(inED_chs, condED, kernel_size=3, stride=1, padding=1),
                act_inED,
            )

    def forward(self, x, rm=None, ed=None):
        if rm is None and ed is None:
            return self.inx(x)
        else:
            fea_x = self.inx(x)
            if rm is not None and ed is not None:
                fea_sub = torch.cat((self.inRM(rm), self.inED(ed)), 1)
            if rm is not None and ed is None:
                fea_sub = self.inRM(rm)
            if rm is None and ed is not None:
                fea_sub = self.inED(ed)
            return fea_x, fea_sub

class mainBlock(nn.Module):
    def __init__(self, nfeat=32, nDlayer=4, grRate=32, dila1=[1,1,1,1], dila2=[1,1,1,1], block_type='U_D',
                 auam=None):
        super(mainBlock, self).__init__()
        self.block_type = block_type
        if block_type == 'U_D':
            dila1 = [2, 4, 8, 16]
            dila2 = [3, 6, 12, 24]
        elif block_type == 'U':
            dila1 = [1, 1, 1, 1]
            dila2 = [1, 1, 1, 1]
        if auam is not None:
            self.inc = nn.Sequential(
                auam,
                DRDB(nfeat, nDlayer, grRate),
            )
            self.conv4 = nn.Sequential(
                auam,
                DRDB(nfeat, nDlayer, grRate),
            )
        else:
            self.inc = nn.Sequential(
                DRDB(nfeat, nDlayer, grRate),
            )
            self.conv4 = nn.Sequential(
                DRDB(nfeat, nDlayer, grRate),
            )
        self.conv1 = nn.Sequential(
            single_conv(nfeat, nfeat * 2),
            DRDB(nfeat * 2, nDlayer, grRate),
        )
        self.conv2 = nn.Sequential(
            single_conv(nfeat * 2, nfeat * 4),
            DRDB(nfeat * 4, nDlayer, grRate),
        )
        self.dila1 = DRDB(nfeat * 4, nDlayer, grRate, dilations=dila1)
        self.dila2 = DRDB(nfeat * 4, nDlayer, grRate, dilations=dila2)
        self.conv_1 = nn.Conv2d(nfeat * 4 * 3, nfeat * 4, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Sequential(
            DRDB(nfeat * 2, nDlayer, grRate),
        )

        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(2)
        self.up1 = up(nfeat * 4)
        self.up2 = up(nfeat * 2)
        self.outc = outconv(nfeat, nfeat)

    def forward(self, x, cond=None):
        # x[0]: fea; x[1]: cond
        if cond is not None:
            inx = self.inc((x, cond))
        else:
            inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)
        # dilation
        dila1 = self.dila1(conv2)
        dila2 = self.dila2(dila1)
        conv2 = self.conv_1(torch.cat([conv2, dila1, dila2], 1))

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        if cond is not None:
            conv4 = self.conv4((up2, cond))
        else:
            conv4 = self.conv4(up2)

        out = self.outc(conv4)

        return out

class AuAM(nn.Module):
    def __init__(self, nfeat=32, act_type='DAF'):
        super(AuAM, self).__init__()
        if act_type == 'DAF':
            self.act = nn.Sequential(
                nn.Conv2d(nfeat, nfeat, 1),
                DAF(nfeat, nfeat * 2, dilations=[4]),
                nn.Conv2d(nfeat * 2, nfeat * 2, 1),
            )
        elif act_type == 'XU':
            self.act = nn.Sequential(
                nn.Conv2d(nfeat, nfeat , 1),
                Modulecell(nfeat, nfeat * 2),
                nn.Conv2d(nfeat * 2, nfeat * 2, 1),
            )
        elif act_type == 'ReLU':
            self.act = nn.Sequential(
                nn.Conv2d(nfeat, nfeat, 1),
                nn.ReLU(),
                nn.Conv2d(nfeat, nfeat * 2, 1),
                nn.ReLU(),
            )

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        return x[0] * (self.act(x[1]) + 1)

class DAF(nn.Module):
    def __init__(self,in_channels=16,out_channels=16,kernel_size=3,skernel_size=9,dilations=None):
        super(DAF, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2)))
        self.module = nn.Sequential(
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=skernel_size,stride=1,padding=((skernel_size-1)//2),groups=out_channels),
            DRDB(out_channels, len(dilations), out_channels, dilations=dilations),)

    def forward(self,x):
        x1 = self.features(x)
        x2 = torch.exp(self.module(x1))
        x = torch.mul(x1,x2)
        return x

class DRDB(nn.Module):
    def __init__(self, nChannels,nDenselayer, growthRate, dilations=[1,1,1,1], nCout=None, ):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        if nCout is None:
            nCout = nChannels
        modules = []
        for i in range(nDenselayer):
            if i < len(dilations):
                dilation = dilations[i]
            else:
                dilation = 1
            modules.append(make_dense(nChannels_, growthRate , dilation=dilation))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nCout, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        if out.shape[1] == x.shape[1]:
            out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, sn=False, dilation=1):
        super(make_dense, self).__init__()
        if dilation == 1:
            pad = (kernel_size - 1) // 2
        else:
            pad = dilation
        if sn:
            self.conv = spectral_norm(
                nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=pad,
                          dilation=dilation,bias=False))
        else:
            self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=pad,
                                  dilation=dilation,bias=False)

    def forward(self, x):
        out = F.leaky_relu(self.conv(x),0.1,inplace=True)
        out = torch.cat((x, out), 1)
        return out

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sn=False):
        super(outconv, self).__init__()
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, sn=False):
        super(single_conv, self).__init__()
        if sn:
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                nn.LeakyReLU(0.1,inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.1,inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # x is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x

# XUnit, Code adapted from: 'https://github.com/kligvasser/xUnit'
class Gaussian(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))

class Modulecell(nn.Module):
    def __init__(self,in_channels=1,out_channels=64,kernel_size=3,skernel_size=9):
        super(Modulecell,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2)))
        self.module = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=skernel_size,stride=1,padding=((skernel_size-1)//2),groups=out_channels),
            nn.BatchNorm2d(out_channels),
            Gaussian())
    def forward(self,x):
        x1 = self.features(x)
        x2 = self.module(x1)
        x = torch.mul(x1,x2)
        return x
