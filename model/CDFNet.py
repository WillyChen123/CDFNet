# coding:utf-8

import torch
import torch.nn as nn 
import torchvision.models as models
from .customize import StripPooling

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class SEAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes,reduction=8):
        super(SEAdaIN,self).__init__()        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_nc, input_nc // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_nc // reduction, input_nc, bias=False),
            nn.Sigmoid()
        )
        self.norm = norm(planes)
    
    def forward(self,x,target):
        size = x.size()
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)

        x_mean, x_std = calc_mean_std(x)
        target_mean, target_std = calc_mean_std(target)

        normalized_feat = (x - x_mean.expand(
            size)) / x_std.expand(size)
        return normalized_feat * target_std.expand(size) + target_mean.expand(size)

class CDFNet(nn.Module):

    def __init__(self, n_class):
        super(CDFNet, self).__init__()

        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########
 
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
 
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4
        
        ###### SEAdaIN module

        self.SEAdaIN_rgb2thermal_1 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=64,planes=64)
        self.SEAdaIN_thermal2rgb_1 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=64,planes=64)
        self.SEAdaIN_rgb2thermal_2 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=256,planes=256)
        self.SEAdaIN_thermal2rgb_2 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=256,planes=256)
        self.SEAdaIN_rgb2thermal_3 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=512,planes=512)
        self.SEAdaIN_thermal2rgb_3 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=512,planes=512)
        self.SEAdaIN_rgb2thermal_4 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=1024,planes=1024)
        self.SEAdaIN_thermal2rgb_4 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=1024,planes=1024)
        self.SEAdaIN_rgb2thermal_5 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=2048,planes=2048)
        self.SEAdaIN_thermal2rgb_5 = SEAdaIN(norm=nn.InstanceNorm2d,input_nc=2048,planes=2048)
        
        ###### Strip pooling module

        self.strip_pool1 = StripPooling(2048, (10, 5), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        self.strip_pool2 = StripPooling(2048, (10, 5), nn.BatchNorm2d, {'mode': 'bilinear', 'align_corners': True})
        
        ########  DECODER  ########

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)
        
        
 
    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
 
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
 
    def forward(self, input):

        rgb = input[:,:3]
        thermal = input[:,3:]

        vobose = False

        # encoder

        ######################################################################

        if vobose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if vobose: print("thermal.size() original: ", thermal.size()) # (480, 640)

        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        if vobose: print("rgb.size() after conv1: ", rgb.size()) # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if vobose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if vobose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        if vobose: print("thermal.size() after conv1: ", thermal.size()) # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if vobose: print("thermal.size() after bn1: ", thermal.size()) # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if vobose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)

        ### SEAdaIN
        rgb_to_thermal1 = self.SEAdaIN_rgb2thermal_1(rgb, thermal)
        thermal_to_rgb1 = self.SEAdaIN_thermal2rgb_1(thermal, rgb)
        
        rgb = rgb + thermal_to_rgb1
        thermal = thermal + rgb_to_thermal1

        rgb = self.encoder_rgb_maxpool(rgb)
        if vobose: print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

        thermal = self.encoder_thermal_maxpool(thermal)
        if vobose: print("thermal.size() after maxpool: ", thermal.size()) # (120, 160)

        ######################################################################
        rgb = self.encoder_rgb_layer1(rgb)
        if vobose: print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
        thermal = self.encoder_thermal_layer1(thermal)
        if vobose: print("thermal.size() after layer1: ", thermal.size()) # (120, 160)

        ### SEAdaIN
        rgb_to_thermal2 = self.SEAdaIN_rgb2thermal_2(rgb, thermal)
        thermal_to_rgb2 = self.SEAdaIN_thermal2rgb_2(thermal, rgb)
        
        rgb = rgb + thermal_to_rgb2
        thermal = thermal + rgb_to_thermal2

        ######################################################################
        rgb = self.encoder_rgb_layer2(rgb)
        if vobose: print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        if vobose: print("thermal.size() after layer2: ", thermal.size()) # (60, 80)

        ### SEAdaIN
        rgb_to_thermal3 = self.SEAdaIN_rgb2thermal_3(rgb, thermal)
        thermal_to_rgb3 = self.SEAdaIN_thermal2rgb_3(thermal, rgb)
        
        
        rgb = rgb + thermal_to_rgb3
        thermal = thermal + rgb_to_thermal3

        ######################################################################
        rgb = self.encoder_rgb_layer3(rgb)
        if vobose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        if vobose: print("thermal.size() after layer3: ", thermal.size()) # (30, 40)

        ### SEAdaIN
        rgb_to_thermal4 = self.SEAdaIN_rgb2thermal_4(rgb, thermal)
        thermal_to_rgb4 = self.SEAdaIN_thermal2rgb_4(thermal, rgb)
        
        
        rgb = rgb + thermal_to_rgb4
        thermal = thermal + rgb_to_thermal4

        ######################################################################
        rgb = self.encoder_rgb_layer4(rgb)
        if vobose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        if vobose: print("thermal.size() after layer4: ", thermal.size()) # (15, 20)
        
        ### SEAdaIN
        rgb_to_thermal5 = self.SEAdaIN_rgb2thermal_5(rgb, thermal)
        thermal_to_rgb5 = self.SEAdaIN_thermal2rgb_5(thermal, rgb)
        
        rgb = rgb + thermal_to_rgb5
        thermal = thermal + rgb_to_thermal5

        fuse = rgb + thermal
        
        fuse = self.strip_pool1(fuse)
        fuse = self.strip_pool2(fuse)
        
        ######################################################################

        # decoder

        fuse = self.deconv1(fuse)
        if vobose: print("fuse after deconv1: ", fuse.size()) # (30, 40)
        fuse = self.deconv2(fuse)
        if vobose: print("fuse after deconv2: ", fuse.size()) # (60, 80)
        fuse = self.deconv3(fuse)
        if vobose: print("fuse after deconv3: ", fuse.size()) # (120, 160)
        fuse = self.deconv4(fuse)
        if vobose: print("fuse after deconv4: ", fuse.size()) # (240, 320)
        fuse = self.deconv5(fuse)
        if vobose: print("fuse after deconv5: ", fuse.size()) # (480, 640)

        return fuse

class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    cdf_net = CDFNet(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    cdf_net(input)
    #print('The model: ', rtf_net.modules)

if __name__ == '__main__':
    unit_test()
