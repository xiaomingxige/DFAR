import torch
import torch.nn as nn
import torch.nn.functional as F


from .darknet import BaseConv, get_activation
from nets.ops.dcn.deform_conv import ModulatedDeformConv


# from darknet import BaseConv, get_activation
# from ops.dcn.deform_conv import ModulatedDeformConv


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)


class HFFB(nn.Module):
    def __init__(self, nc=64):
        super(HFFB, self).__init__()
        nf = nc // 2
        self.c1 = conv_layer(nc, nf, 3, 1, 1)
        self.d1 = conv_layer(nf, nf, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nf, nf, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nf, nf, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nf, nf, 3, 1, 4)  # rate=4
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c2 = conv_layer(nc + nf * 4, nc, 1, 1, 1)  # 256-->64

        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.ca_block = CA_block(in_channel=nc + nc // 2 * 4, reduce_ratio=2)

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4

        combine = torch.cat([input, d1, add1, add2, add3], 1)
        
        combine = self.ca_block(combine)

        output2 = self.c2(self.act(combine))
        return input * self.fuse_weight_1 + output2 * self.fuse_weight_2

    


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



class Feature_Extractor(nn.Module):   
    def __init__(self, in_nc=3, nf=64, out_nc=64, act="relu"):
        super(Feature_Extractor, self).__init__()
        self.act = get_activation(act, inplace=True)

        self.down_conv1 = nn.Sequential(
            BaseConv(in_nc, nf, 3, 1, act=act),
            BaseConv(nf, nf, 3, 2, act=act),
        )

        self.down_conv2 = nn.Sequential(
            BaseConv(nf, nf, 3, 1, act=act),
            BaseConv(nf, nf, 3, 2, act=act),
        )

        self.down_conv3 = nn.Sequential(
            BaseConv(nf, nf, 3, 1, act=act),
            BaseConv(nf, nf, 3, 2, act=act),
        )

        self.out_conv = nn.Sequential(
            BaseConv(nf, out_nc, 3, 1, act=act),
        )
    
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        return self.out_conv(x3)
    

class Align_Net(nn.Module):   
    def __init__(self, in_nc, out_nc, base_ks=3, deform_nc=64, deform_ks=3, deform_group=8):
        super(Align_Net, self).__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.deform_group = deform_group
        self.offset_mask = nn.Conv2d(in_nc, deform_group*3*(deform_ks**2), base_ks, padding=base_ks//2)
        self.deform_conv = ModulatedDeformConv(deform_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=deform_group)
    
    def forward(self, x, y):
        off_msk = self.offset_mask(x)
        off = off_msk[:, :self.deform_group*2*(self.deform_ks**2), ...]
        msk = torch.sigmoid(off_msk[:, self.deform_group*2*(self.deform_ks**2):, ...])
        fused_feat = F.relu(self.deform_conv(y, off, msk), inplace=True)
        return fused_feat
    

    
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


class DSTA(nn.Module):
    def __init__(self, n_feats, act):
        super(DSTA, self).__init__()
        f = n_feats // 2
        self.conv1 = BaseConv(n_feats, f, 3, 1, act=act)

        self.spatial_attention = Spatial_Attention(out_nc=f, kernel_size=7)
        self.channel_attention = Channel_Attention(f, ratio=2)
        self.fuse_layer = BaseConv(2 * f, f, 1, 1, act=act)

        self.mask1 = default_conv(f, f*3*3*3, 3, padding=1)
        self.down_conv1 = BaseConv(f, f, 3, 2, act=act)
        self.mask2 = default_conv(f, f*3*3*3, 3, padding=1)

        self.f = f
        self.dcn = ModulatedDeformConv(f, f, 3, padding=1, deformable_groups=f)
        self.out_conv = BaseConv(f, n_feats, 3, 1, act=act)

    def forward(self, x):
        x2 = self.conv1(x)

        x2_s = self.spatial_attention(x2)
        x2_c = self.channel_attention(x2)
        x2 = torch.cat((x2_s, x2_c), dim=1)
        x2 = self.fuse_layer(x2)
        off_mask1 = self.mask1(x2)

        x3 = self.down_conv1(x2)
        off_mask2 = self.mask2(x3)
        off_mask2 = F.interpolate(off_mask2, (off_mask1.size(2), off_mask1.size(3)), mode='bilinear', align_corners=False)
        off_mask1 = off_mask1 + off_mask2
        off = off_mask1[:, :self.f*2*3*3, ...]
        mask = torch.sigmoid(off_mask1[:, self.f*2*3*3:, ...])

        out = F.relu(self.dcn(x2, off, mask), inplace=True)
        out = self.out_conv(out)
        return out


class Fuse_Net(nn.Module):
    def __init__(self, in_fea, out_fea, branches=5, reduce=4, len=32):
        super(Fuse_Net, self).__init__()
        self.branches = branches

        len = max(in_fea // reduce, len)

        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels=in_fea * branches, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(out_fea, len, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, out_fea, kernel_size=1, stride=1)
            )
        # self.softmax = nn.Softmax(dim=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_fea * branches, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, out):
        out = torch.stack(out, dim=1)
        b, t, c, h, w = out.shape
        attention = out.view(b, -1, h, w)
        attention = self.conv_attention(attention)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        out = out * attention  # b, 3, c, h, w

        out = out.view(b, -1, h, w)
        out = self.conv(out)
        return out


import functools
class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5, training=True):
        super(Network, self).__init__()
        self.num_frame = num_frame
        # act = 'relu'  # silu
        act = 'silu'  # silu

        fea_ext_nf = 48
        fea_ext_out_nc = 64
        self.fea_ext = Feature_Extractor(in_nc=3, nf=fea_ext_nf, out_nc=fea_ext_out_nc, act=act) 


        self.fuse_layers = nn.ModuleList([])
        self.align_layers = nn.ModuleList([])
        for i in range(4):
            self.fuse_layers.append(nn.Sequential(
                BaseConv(fea_ext_out_nc*2, fea_ext_out_nc, 3, 1, act=act),
                self.make_layer(functools.partial(HFFB, fea_ext_out_nc), 4), 
                BaseConv(fea_ext_out_nc, 64, 3, 1, act=act),
            ))
            self.align_layers.append(
                Align_Net(64, fea_ext_out_nc, 3, fea_ext_out_nc, 3, 8)
            )

        self.ada_fuse = Fuse_Net(in_fea=fea_ext_out_nc, out_fea=fea_ext_out_nc, branches=num_frame, reduce=2, len=32)

        fine_align_in_nc = 64
        self.mid_conv = nn.Sequential(
            BaseConv(fea_ext_out_nc, fine_align_in_nc, 3, 1, act=act)
        )

        self.dsat_layers = self.make_layer(functools.partial(DSTA, fine_align_in_nc, act=act), 4)

        self.head_nf = 128
        self.out_conv = nn.Sequential(
            BaseConv(fine_align_in_nc, self.head_nf, 3, 1, act=act)
        )
        
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[self.head_nf])

        self.loss_function = nn.L1Loss()
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        feat = []
        for i in range(self.num_frame):
            feat.append(self.fea_ext(inputs[:, :, i, :, :]))

        ref_feat = feat[2]
        align_feat_list = []
        for i in range(self.num_frame):
            if i == 2:
                align_feat_list.append(ref_feat)
                continue
            else:
                fuse_feat = torch.cat((feat[i], ref_feat), dim=1)
                if i < 2 :
                    fuse_feat = self.fuse_layers[i](fuse_feat)
                    align_feat = self.align_layers[i](fuse_feat, feat[i])
                else:
                    fuse_feat = self.fuse_layers[i-1](fuse_feat)
                    align_feat = self.align_layers[i-1](fuse_feat, feat[i])

                if self.training:
                    if i == 0:
                        motion_loss = self.loss_function(align_feat, ref_feat)
                    else:
                        motion_loss += self.loss_function(align_feat, ref_feat)

                align_feat_list.append(align_feat)

        out_feat = self.ada_fuse(align_feat_list)
        out_feat = self.mid_conv(out_feat)
        out_feat = self.dsat_layers(out_feat)
        out_feat = self.out_conv(out_feat)
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
               

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)  
