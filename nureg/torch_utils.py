import numpy as np
from copy import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd

def to_variable(x, requires_grad=False, cuda=False, var=True):
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad)
    if cuda:
       return x.cuda()
    else:
       return x

def to_device(src, ref, var = True, requires_grad=False):

    src = to_variable(src, requires_grad=requires_grad, var=var)
    return src.cuda(ref.get_device()) if ref.is_cuda else src
