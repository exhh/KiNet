import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.nn.functional as F

from ..torch_utils import to_device
from ..torch_utils import Indexflow
from ..util import safe_load_state_dict
from .models import register_model
import functools
import numpy as np

__all__ = ['Ki67Net', 'ki67net']

def passthrough(x, **kwargs):
    return x

def convAct(nchan):
    #return nn.PReLU(nchan)
    return nn.ELU(inplace=True)

class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

class ConvBN(nn.Module):
    def __init__(self, nchan, inChans=None):
        super(ConvBN, self).__init__()
        if inChans is None:
            inChans = nchan
        self.act = convAct(nchan)
        self.conv = nn.Conv2d(inChans, nchan, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(nchan)

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        return out

def _make_nConv(nchan, depth):
    layers = []
    if depth >=0:
        for _ in range(depth):
            layers.append(ConvBN(nchan))
        return nn.Sequential(*layers)
    else:
        return passthrough

class InputTransition(nn.Module):
    def __init__(self,inputChans, outChans):
        self.outChans = outChans
        self.inputChans = inputChans
        super(InputTransition, self).__init__()
        self.conv = nn.Conv2d(inputChans, outChans, kernel_size=3, padding=1)
        self.bn = ContBatchNorm2d(outChans)
        self.relu = convAct(outChans)

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.inputChans == 1:
            x_aug = torch.cat([x]*self.outChans, 0)
            out = self.relu(torch.add(out, x_aug))
        else:
            out = self.relu(out)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        #outChans = 2*inChans
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, stride=2)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

def match_tensor(out, refer_shape):
    skiprow,skipcol = refer_shape
    row, col = out.size()[2], out.size()[3]
    if skipcol >= col:
        pad_col = skipcol - col
        left_pad_col  = pad_col // 2
        right_pad_col = pad_col - left_pad_col
        out = F.pad(out, (left_pad_col, right_pad_col, 0,0))
    else:
        crop_col = col - skipcol
        left_crop_col  = crop_col // 2
        right_col = left_crop_col + skipcol
        out = out[:,:,:, left_crop_col:right_col]

    if skiprow >= row:
        pad_row = skiprow - row
        left_pad_row  = pad_row // 2
        right_pad_row = pad_row - left_pad_row
        out = F.pad(out, (0,0, left_pad_row,right_pad_row))
    else:
        crop_row = row - skiprow
        left_crop_row  = crop_row // 2

        right_row = left_crop_row + skiprow

        out = out[:,:,left_crop_row:right_row, :]

    return out


class UpConcat(nn.Module):
    def __init__(self, inChans, hidChans, outChans, nConvs, dropout=False,stride=2):
        # remeber inChans is mapped to hidChans, then concate together with skipx, the mixed channel =outChans
        super(UpConcat, self).__init__()
        #hidChans = outChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, hidChans, kernel_size=3,
                                          padding=1, stride=stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(hidChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = convAct(hidChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, skipxdo.size()[2:])

        xcat = torch.cat([out, skipxdo], 1)
        out  = self.ops(xcat)
        out  = self.relu2(out + xcat)
        return out

class UpConv(nn.Module):
    def __init__(self, inChans, outChans, nConvs,dropout=False, stride = 2):
        super(UpConv, self).__init__()
        #hidChans = outChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=3,
                                          padding=1, stride = stride, output_padding=1)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

    def forward(self, x, dest_size):
        '''
        dest_size should be (row, col)
        '''
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        out = match_tensor(out, dest_size)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans,outChans=1,hidChans=2):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(inChans, hidChans, kernel_size=5, padding=2)
        self.bn1   = ContBatchNorm2d(hidChans)
        self.relu1 = convAct( hidChans)
        self.conv2 = nn.Conv2d(hidChans, outChans, kernel_size=1)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

class ResdualBlock_KiNet(nn.Module):
    def __init__(self, inChans, outChans, nConvs=1, dropout=False):
        super(ResdualBlock_KiNet, self).__init__()
        self.down_conv = nn.Conv2d(inChans, outChans, kernel_size=1)
        self.bn1 = ContBatchNorm2d(outChans)
        self.do1 = passthrough
        self.relu1 = convAct(outChans)
        self.relu2 = convAct(outChans)
        if dropout:
            self.do1 = nn.Dropout2d()

        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(out+down)
        return out

class Ki67Net(nn.Module):
    def __init__(self, nll=False):
        super(Ki67Net, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.in_tr_100     = InputTransition(3, 32)
        self.down_tr32_50  = DownTransition(32, 64, 1)
        self.down_tr64_25  = DownTransition(64, 128, 2)
        self.down_tr128_12 = DownTransition(128, 256, 2,dropout=True)
        self.down_tr256_6  = DownTransition(256, 256, 2,dropout=True)

        self.up_tr256_12   = UpConcat(256, 256, 512, 2,  dropout=True)
        self.up_tr128_25   = UpConcat(512, 128, 256, 2, dropout=True)
        self.up_tr64_50    = UpConcat(256, 64, 128, 1)
        self.up_tr32_100   = UpConcat(128, 32, 64, 1)

        self.up_12_100   = UpConv(512, 64, 2, stride = 8)
        self.up_25_100   = UpConv(256, 64, 2, stride = 4)
        self.out_tr      = OutputTransition(64*3, 3, 32)

    def forward(self, x):
        x = to_device(x,self.device_id)
        out16 = self.in_tr_100(x)
        out32 = self.down_tr32_50(out16)
        out64 = self.down_tr64_25(out32)
        out128 = self.down_tr128_12(out64)
        out256 = self.down_tr256_6(out128)

        out_up_12 = self.up_tr256_12(out256, out128)
        out_up_25 = self.up_tr128_25(out_up_12, out64)
        out_up_50 = self.up_tr64_50(out_up_25, out32)
        out_up_50_100 = self.up_tr32_100(out_up_50, out16)

        out_up_12_100 = self.up_12_100(out_up_12, x.size()[2:])
        out_up_25_100 = self.up_25_100(out_up_25, x.size()[2:])
        out_cat = torch.cat([out_up_50_100,out_up_12_100, out_up_25_100 ], 1)
        out = self.out_tr(out_cat)

        return out

    def predict(self, batch_data, batch_size=None):
        self.eval()
        total_num = batch_data.shape[0]
        if batch_size is None or batch_size >= total_num:
            x = to_device(batch_data, self.device_id, False).float()
            det = self.forward(x)
            return det.cpu().data.numpy()
        else:
            results = []
            for ind in Indexflow(total_num, batch_size, False):
                data = batch_data[ind]
                data = to_device(data, self.device_id, False).float()
                det = self.forward(data)
                results.append(det.cpu().data.numpy())
            return np.concatenate(results,axis=0)

@register_model('ki67net')
def ki67net(pretrained=True, finetune=False, out_map=True, **kwargs):
    model = Ki67Net()
    return model
