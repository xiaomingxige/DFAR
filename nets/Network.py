import torch
import torch.nn as nn
import torch.nn.functional as F


from .darknet import BaseConv, get_activation
from nets.ops.dcn.deform_conv import ModulatedDeformConv



class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.stems      = nn.ModuleList()

        self.cls_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
    
        self.reg_convs  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()

        self.obj_preds  = nn.ModuleList()
        headnf = int(256 * width)

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = headnf, ksize = 1, stride = 1, act = act))
            
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):   # B, C, H, W
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)

            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)  # cls_output: B, num_classes, H, W

            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)  # reg_output: B, 4, H, W

            obj_output  = self.obj_preds[k](reg_feat)  # cls_output: B, 1, H, W

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs




def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)


class CA_block(nn.Module):   
    def __init__(self, in_channel=32, reduce_ratio=4):
        super(CA_block, self).__init__()
        self.ca_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduce_ratio, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channel // reduce_ratio, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.ca_layer(x)
        x = x * x1
        return x



    
class Spatial_Attention(nn.Module):
    def __init__(self, out_nc=1, kernel_size=7):
        super(Spatial_Attention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, out_nc, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        y = maxout.view(2, -1)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x


class Channel_Attention(nn.Module):
    def __init__(self, in_nc, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_nc, in_nc // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_nc // ratio, in_nc, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x
    
    
def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
       if not padding and stride==1:
           padding = kernel_size // 2
       return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)  


import functools
class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5, training=True):
        super(Network, self).__init__()
        self.num_frame = num_frame
        act = 'silu'  # silu




        fea_ext_nf = 48
        fea_ext_out_nc = 64
        # Thanks for your attention! After the paper accept, we will open the details soon.
        
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[self.head_nf])

        self.loss_function = nn.L1Loss()
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        feat = []
        for i in range(self.num_frame):
            feat.append(self.fea_ext(inputs[:, :, i, :, :]))


        # Thanks for your attention! After the paper accept, we will open the details soon.
        outputs = self.head([out_feat])

        if self.training:
            return  outputs, motion_loss  
        else:
            return  outputs
    
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)