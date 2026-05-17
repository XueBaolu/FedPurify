'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
    


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCifar_SD(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None):
        super(ResNetCifar_SD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "\
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                    dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                    dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                    dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        self.scala = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # use (4,4) for cifar-10 and 100 and (8,8) for tiny_imagenet
            nn.AvgPool2d(4, 4),  
        )
        
        self.attention = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, weights = None, get_feat=None, SD = None):
        # See note [TorchScript super()]
        if weights == None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            feat1 = self.layer1(x)
            feat2 = self.layer2(feat1)
            
            feat3 = self.layer3(feat2)
            feat4 = self.layer4(feat3)
            
            if SD != None:
                atten = self.attention(feat2)
                feat2 = atten*feat2
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc2(SD_feat)
            
            x = self.avgpool(feat4)
            feat = torch.flatten(x, 1)
            x = self.fc(feat)
            
            
            if get_feat == None and SD ==None:
                return x
            elif get_feat != None and SD ==None:
                return x, feat
            elif get_feat != None and SD != None:
                return x, SD_x, feat 
            elif get_feat == None and SD != None:
                return x, SD_x 
            
        else:     
            x = F.conv2d(x, weights['conv1.weight'], bias=None, stride=1, padding=1)
            x = F.batch_norm(x, self.bn1.running_mean, self.bn1.running_var, weights['bn1.weight'], weights['bn1.bias'],training=True)            
            x = F.relu(x, inplace=True)
            #layer 1
            for i in range(2):
                residual = x
                out = F.conv2d(x, weights['layer1.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn1.running_mean, self.layer1[i].bn1.running_var, 
                                weights['layer1.%d.bn1.weight'%i], weights['layer1.%d.bn1.bias'%i],training=True)      
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer1.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer1[i].bn2.running_mean, self.layer1[i].bn2.running_var, 
                                weights['layer1.%d.bn2.weight'%i], weights['layer1.%d.bn2.bias'%i],training=True)   
                out = F.relu(out, inplace=True)                         
                x = out + residual     
                x = F.relu(x, inplace=True)
                feat1 = x

            #layer 2
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer2.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn1.running_mean, self.layer2[i].bn1.running_var, 
                                weights['layer2.%d.bn1.weight'%i], weights['layer2.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer2.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer2[i].bn2.running_mean, self.layer2[i].bn2.running_var, 
                                weights['layer2.%d.bn2.weight'%i], weights['layer2.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer2.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer2[i].downsample[1].running_mean, self.layer2[i].downsample[1].running_var, 
                                weights['layer2.%d.downsample.1.weight'%i], weights['layer2.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat2 = x

            #layer 3
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer3.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn1.running_mean, self.layer3[i].bn1.running_var, 
                                weights['layer3.%d.bn1.weight'%i], weights['layer3.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer3.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer3[i].bn2.running_mean, self.layer3[i].bn2.running_var, 
                                weights['layer3.%d.bn2.weight'%i], weights['layer3.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer3.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer3[i].downsample[1].running_mean, self.layer3[i].downsample[1].running_var, 
                                weights['layer3.%d.downsample.1.weight'%i], weights['layer3.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat3 = x
                
            #layer 4
            for i in range(2):
                residual = x
                if i == 0:
                    out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], bias=None, stride=2, padding=1)
                else:
                    out = F.conv2d(x, weights['layer4.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn1.running_mean, self.layer4[i].bn1.running_var, 
                                weights['layer4.%d.bn1.weight'%i], weights['layer4.%d.bn1.bias'%i],training=True)     
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['layer4.%d.conv2.weight'%i], bias=None, stride=1, padding=1)
                out = F.batch_norm(out, self.layer4[i].bn2.running_mean, self.layer4[i].bn2.running_var, 
                                weights['layer4.%d.bn2.weight'%i], weights['layer4.%d.bn2.bias'%i],training=True)    
                if i==0:
                    residual = F.conv2d(x, weights['layer4.%d.downsample.0.weight'%i], bias=None, stride=2)  
                    residual = F.batch_norm(residual, self.layer4[i].downsample[1].running_mean, self.layer4[i].downsample[1].running_var, 
                                weights['layer4.%d.downsample.1.weight'%i], weights['layer4.%d.downsample.1.bias'%i],training=True)  
                x = out + residual  
                x = F.relu(x, inplace=True)
            feat4 = x
                
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            feat = x.view(x.size(0), -1)
            x = F.linear(feat, weights['fc.weight'], weights['fc.bias'])    
            
            if SD == None:
                return x
            else:
                atten = self.attention(feat2)
                feat2 = atten*feat2
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc2(SD_feat)
                
                return x, SD_x        


from .wrn import BasicBlock, NetworkBlock

def ResNet18_SD_cifar(class_num, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNetCifar_SD(BasicBlock, [2, 2, 2, 2], num_classes = class_num, **kwargs)


class SimpleCNN_SD(nn.Module):
    def __init__(self, output_dim=64, n_classes=10, width=2):
        super(SimpleCNN_SD, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16*width, kernel_size=3, bias=False) 
        self.conv1_bn = nn.BatchNorm2d(16*width)
        self.conv2 = nn.Conv2d(16*width, 32*width, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(32*width)        
        
        self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(32*width, output_dim)
        self.fc1_bn = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, n_classes)
        
        sepcov_dim = 64
        self.attention = nn.Sequential(
            SepConv(
                channel_in=sepcov_dim,
                channel_out=sepcov_dim
            ),
            nn.BatchNorm2d(sepcov_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        self.scala = nn.Sequential(
            SepConv(
                channel_in=sepcov_dim,
                channel_out=2*sepcov_dim,
            ),
            SepConv(
                channel_in=2*sepcov_dim,
                channel_out=4*sepcov_dim,
            ),
            # use (4,4) for cifar-10 and 100 and (8,8) for tiny_imagenet
            nn.AvgPool2d(4, 4),  
        )
        self.fc_ext = nn.Linear(4*sepcov_dim, n_classes)

    def forward(self, x, weights = None, get_feat=None, SD=None):
        if weights == None:
            x = self.conv1(x)
            x = self.conv1_bn(x)
            x = self.relu(x)
            feat1 = self.conv2(x)
            feat1 = self.conv2_bn(feat1)
            feat1 = self.relu(feat1)
            
            if SD != None:
                atten = self.attention(feat1)
                feat2 = feat1 * atten
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc_ext(SD_feat)
            
            x = self.avgpool(feat1)
            feat = torch.flatten(x, 1)
            x = self.fc1(feat)
            x = self.fc1_bn(x)
            x = self.relu(x)
            y = self.fc2(x)  
            
            if get_feat == None and SD == None:
                return y
            elif get_feat != None and SD == None:
                return y, x
            elif get_feat != None and SD != None:
                return y, SD_x, x 
            elif get_feat == None and SD != None:
                return y, SD_x
        
        else:
            x = F.conv2d(x, weights['conv1.weight'], bias=None, stride=1, padding=1)
            x = F.batch_norm(x, self.conv1_bn.running_mean, self.conv1_bn.running_var, \
                weights['conv1_bn.weight'], weights['conv1_bn.bias'],training=True)            
            x = F.relu(x, inplace=True)
            
            feat1 = F.conv2d(x, weights['conv2.weight'], bias=None, stride=1, padding=0)
            feat1 = F.batch_norm(feat1, self.conv2_bn.running_mean, self.conv2_bn.running_var, \
                weights['conv2_bn.weight'], weights['conv2_bn.bias'],training=True)      
            feat1 = F.relu(feat1, inplace=True)
            
            x = F.adaptive_avg_pool2d(feat1, output_size=(1, 1))
            feat = torch.flatten(x, 1)
            x = F.linear(feat, weights['fc1.weight'], weights['fc1.bias'])
            x = F.batch_norm(x, self.fc1_bn.running_mean, self.fc1_bn.running_var,\
                weights['fc1_bn.weight'], weights['fc1_bn.bias'],training=True)
            x = F.relu(x, inplace=True)
            y = F.linear(x, weights['fc2.weight'], weights['fc2.bias'])
        
            if SD == None:
                return y
            else:
                atten = self.attention(feat1)
                feat2 = feat1 * atten
                SD_x = self.scala(feat2)
                SD_feat = torch.flatten(SD_x, 1)
                SD_x = self.fc_ext(SD_feat)
                return y, SD_x


class WRN_SD(nn.Module):
    # Wide ResNet SD model for CIFAR-10
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WRN_SD, self).__init__()
        # Implementation details would go here
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                            padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        # for distillation
        sepconv_dim = 64
        self.scala = nn.Sequential(
            SepConv(
                channel_in=sepconv_dim,
                channel_out=sepconv_dim,
            ),
            SepConv(
                channel_in=sepconv_dim,
                channel_out=2 * sepconv_dim,
            ),
            # use (4,4) for cifar-10 and 100 and (8,8) for tiny_imagenet
            nn.AvgPool2d(4, 4),  
        )
        self.fc_ext = nn.Linear(2 * sepconv_dim, num_classes)
        self.attention = nn.Sequential(
            SepConv(
                channel_in=sepconv_dim,
                channel_out=sepconv_dim
            ),
            nn.BatchNorm2d(sepconv_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        

    def forward(self, x, weights=None, get_feat=None, SD=None):
        if weights == None:
            out = self.conv1(x)
            out = self.block1(out)
            feat1 = self.block2(out)
            feat2 = self.block3(feat1)
            feat3 = self.relu(self.bn1(feat2))
            

            if SD != None:
                atten = self.attention(feat1)
                feat2_out = atten * feat1
                SD_x = self.scala(feat2_out)
                SD_feat = torch.flatten(SD_x, 1)
                SD_out = self.fc_ext(SD_feat)
                
            x = F.adaptive_avg_pool2d(feat3, (1,1))
            feat = x.view(-1, self.nChannels)
            out = self.fc(feat)
            
            if get_feat == None and SD == None:
                return out
            elif get_feat != None and SD == None:
                return out, feat
            elif get_feat != None and SD != None:
                return out, SD_out, feat 
            elif get_feat == None and SD != None:
                return out, SD_out 
        
        else:
            # Implement forward pass with external weights if needed
            out = F.conv2d(x, weights['conv1.weight'], bias=None, stride=1, padding=1)
            # block 1
            for i in range(2):
                out = F.batch_norm(out, self.block1.layer[i].bn1.running_mean, self.block1.layer[i].bn1.running_var, 
                                weights['block1.layer.%d.bn1.weight'%i], weights['block1.layer.%d.bn1.bias'%i],training=True)
                out = F.relu(out, inplace=True)
                residual = out
                out = F.conv2d(out, weights['block1.layer.%d.conv1.weight'%i], bias=None, stride=1, padding=1)
                
                out = F.batch_norm(out, self.block1.layer[i].bn2.running_mean, self.block1.layer[i].bn2.running_var, 
                                weights['block1.layer.%d.bn2.weight'%i], weights['block1.layer.%d.bn2.bias'%i],training=True)   
                out = F.relu(out, inplace=True)
                out = F.conv2d(out, weights['block1.layer.%d.conv2.weight'%i], bias=None, stride=1, padding=1)                        
                
                key = f'block1.layer.{i}.convShortcut.weight'
                if key in weights:
                    residual = F.conv2d(residual, weights['block1.layer.%d.convShortcut.weight'%i], bias=None, stride=1, padding=0)
                
                feat1 = torch.add(residual, out)
                # feat1 = F.relu(out, inplace=True)
            
            # block 2
            feat2 = feat1
            for i in range(2):
                feat2 = F.batch_norm(feat2, self.block2.layer[i].bn1.running_mean, self.block2.layer[i].bn1.running_var, 
                                weights['block2.layer.%d.bn1.weight'%i], weights['block2.layer.%d.bn1.bias'%i],training=True)
                feat2 = F.relu(feat2, inplace=True)
                residual = feat2
                feat2 = F.conv2d(feat2, weights['block2.layer.%d.conv1.weight'%i], bias=None, stride=2-i, padding=1)
                # print(feat2.shape)
                feat2 = F.batch_norm(feat2, self.block2.layer[i].bn2.running_mean, self.block2.layer[i].bn2.running_var, 
                                weights['block2.layer.%d.bn2.weight'%i], weights['block2.layer.%d.bn2.bias'%i],training=True)   
                feat2 = F.relu(feat2, inplace=True)
                feat2 = F.conv2d(feat2, weights['block2.layer.%d.conv2.weight'%i], bias=None, stride=1, padding=1)                        
                # print(feat2.shape)
                key = f'block2.layer.{i}.convShortcut.weight'
                if key in weights:
                    residual = F.conv2d(residual, weights['block2.layer.%d.convShortcut.weight'%i], bias=None, stride=2-i, padding=0)
                # print(residual.shape, feat2.shape)
                feat2 = torch.add(residual, feat2)
                # feat2 = F.relu(feat2, inplace=True)
            
            # block 3
            feat3 = feat2
            for i in range(2):
                feat3 = F.batch_norm(feat3, self.block3.layer[i].bn1.running_mean, self.block3.layer[i].bn1.running_var, 
                                weights['block3.layer.%d.bn1.weight'%i], weights['block3.layer.%d.bn1.bias'%i],training=True)
                feat3 = F.relu(feat3, inplace=True)
                residual = feat3
                feat3 = F.conv2d(feat3, weights['block3.layer.%d.conv1.weight'%i], bias=None, stride=2-i, padding=1)
                
                feat3 = F.batch_norm(feat3, self.block3.layer[i].bn2.running_mean, self.block3.layer[i].bn2.running_var, 
                                weights['block3.layer.%d.bn2.weight'%i], weights['block3.layer.%d.bn2.bias'%i],training=True)   
                feat3 = F.relu(feat3, inplace=True)
                feat3 = F.conv2d(feat3, weights['block3.layer.%d.conv2.weight'%i], bias=None, stride=1, padding=1)                        
                
                key = f'block3.layer.{i}.convShortcut.weight'
                if key in weights:
                    residual = F.conv2d(residual, weights['block3.layer.%d.convShortcut.weight'%i], bias=None, stride=2-i, padding=0)
                
                feat3 = torch.add(residual, feat3)
                # feat = F.relu(feat3, inplace=True)
            
            feat3 = F.batch_norm(feat3, self.bn1.running_mean, self.bn1.running_var,\
                weights['bn1.weight'], weights['bn1.bias'],training=True)
            feat3 = F.relu(feat3, inplace=True)
            
            feat = F.adaptive_avg_pool2d(feat3, output_size=(1, 1))
            feat = feat.view(-1, self.nChannels)
            x = F.linear(feat, weights['fc.weight'], weights['fc.bias'])

            if SD == None:
                return x
            else:
                atten = self.attention(feat2)              
                
                feat2_out = atten * feat2
                SD_x = self.scala(feat2_out)
                SD_feat = torch.flatten(SD_x, 1)
                SD_out = self.fc_ext(SD_feat)
                
                return x, SD_out

cfg = {
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG17': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_SD(nn.Module):
    def __init__(self, vgg_name="VGG9", num_classes=100):
        super(VGG_SD, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def get_sd_models(data_name):
    if data_name == 'fmnist':
        global_model = SimpleCNN_SD(n_classes=10)
    elif data_name == 'cifar10':
        global_model = WRN_SD(depth=16, num_classes=10, widen_factor=2, dropout_rate=0.)
    elif data_name == 'tinyimagenet':
        global_model = ResNet18_SD_cifar(200)
    elif data_name == 'cifar100':
        # global_model = VGG_SD()
        global_model = WRN_SD(depth=16, num_classes=100, widen_factor=2, dropout_rate=0.)
    else:
        raise NotImplementedError(f'Model for dataset {data_name} is not implemented.')
    
    return global_model


if __name__ == "__main__":
    batch_size = 32
    num_classes = 10

    x = torch.rand(batch_size, 3, 32, 32)
    y = torch.randint(0, num_classes, (batch_size,))
    
    # model = WRN_SD(depth=16, num_classes=10, widen_factor=2, dropout_rate=0.)
    # global_model = WRN_SD(depth=16, num_classes=10, widen_factor=2, dropout_rate=0.)
    
    x = torch.rand(batch_size, 1, 28, 28)
    model = SimpleCNN_SD(n_classes=10)
    global_model = SimpleCNN_SD(n_classes=10)
    
    outputs, SD_outputs, feat = model(x, get_feat=True, SD=True)
    outputs, SD_outputs = model(x, weights=global_model.state_dict(), get_feat=True, SD=True)