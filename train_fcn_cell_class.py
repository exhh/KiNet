import logging
import os.path, sys, pdb
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
import pdb
from torchvision import transforms
from PIL import Image

from nureg.data.data_loader import get_fcn_dataset as get_dataset
from nureg.models import get_model
from nureg.models.models import models
from nureg.transforms import augment_collate
from nureg.util import config_logging
from nureg.util import to_tensor_raw
from nureg.tools.util import make_variable
from nureg.torch_utils import to_device

try:
    from visdom import Visdom
except:
    print('Better install visdom')


def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def get_weight_mask(y_true, params = None):
    if params is not None:
        y_true = y_true.float() / 255.0 * params['scale']
        mean_label = torch.mean(torch.mean(y_true, dim = -1, keepdim=True), dim=-2, keepdim=True)
        y_mask = y_true / params['scale'] + params['alpha'] * mean_label / params['scale']
    else:
        y_mask = torch.ones(y_true.size())
    return y_true, y_mask

def mean_squared_error(y_true, y_pred, y_mask):
    diff = y_pred - y_true
    naive_loss = diff*diff
    masked =  naive_loss * y_mask
    last_dim = len(y_pred.size()) - 1
    return torch.mean(masked, dim=last_dim)

def weighted_loss(y_true, y_pred, y_mask):
    '''
    y_true and y_pred are (batch, channel, row, col), we need to permite the dimensio first
    '''
    assert y_pred.dim() == 4, 'dimension is not matched!!'
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    y_true = y_true.permute(0,2,3,1)
    y_pred = y_pred.permute(0,2,3,1)
    y_mask = y_mask.permute(0,2,3,1)
    masked_loss = mean_squared_error(y_true,y_pred,y_mask)
    return torch.mean(masked_loss)

def get_validation(datapath, patchsize, imgext='.png'):
    im_dir = os.path.join(datapath, 'images', 'val')
    ids = []
    for filename in os.listdir(im_dir):
        if filename.endswith(imgext):
            ids.append(filename[:-4])
    valid_data = {}
    valid_data['image'] = torch.zeros(len(ids), 3, patchsize, patchsize)
    valid_data['label'] = torch.zeros(len(ids), 3, patchsize, patchsize)
    for idx, im_id in enumerate(ids):
        imagename = os.path.join(datapath, 'images', 'val', im_id+imgext)
        image = Image.open(imagename).convert('RGB')
        labelname_postm = os.path.join(datapath, 'labels_postm', 'val', im_id+'_label.png')
        label_postm = Image.open(labelname_postm).convert('L')
        labelname_negtm = os.path.join(datapath, 'labels_negtm', 'val', im_id+'_label.png')
        label_negtm = Image.open(labelname_negtm).convert('L')
        labelname_other = os.path.join(datapath, 'labels_other', 'val', im_id+'_label.png')
        label_other = Image.open(labelname_other).convert('L')
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(patchsize, patchsize))
        image = TF.crop(image, i, j, h, w)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label_postm = TF.crop(label_postm, i, j, h, w)
        label_postm = torch.from_numpy(np.array(label_postm, np.int64, copy=False)).unsqueeze(0)
        label_negtm = TF.crop(label_negtm, i, j, h, w)
        label_negtm = torch.from_numpy(np.array(label_negtm, np.int64, copy=False)).unsqueeze(0)
        label_other = TF.crop(label_other, i, j, h, w)
        label_other = torch.from_numpy(np.array(label_other, np.int64, copy=False)).unsqueeze(0)
        label = torch.cat((label_postm,label_negtm,label_other), dim=0)

        valid_data['image'][idx,:,:,:] = image
        valid_data['label'][idx,:,:,:] = label

    return valid_data

def display_loss(steps, values, plot=None, name='default', legend= None):
    if plot is None:
        plot = Visdom(use_incoming_socket=False)
    if type(steps) is not list:
        steps = [steps]
    assert type(values) is list, 'values have to be list'
    if type(values[0]) is not list:
        values = [values]

    n_lines = len(values)
    repeat_steps = [steps]*n_lines
    steps  = np.array(repeat_steps).transpose()
    values = np.array(values).transpose()
    win = name + '_loss'
    res = plot.line(
            X = steps,
            Y=  values,
            win= win,
            update='replace',
            opts=dict(title = win)
        )
    if res != win:
        plot.line(
            X = steps,
            Y=  values,
            win=win,
            opts=dict(title = win)
        )


@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='frcn', type=click.Choice(models.keys()))
@click.option('--num_cls', default=1, type=int)
@click.option('--gpu', default='0')
@click.option('--use_validation/--no-use_validation', default=False)
def main(output, dataset, datadir, batch_size, lr, step, iterations,
        momentum, snapshot, downscale, augmentation, use_validation, fyu, crop_size,
        weights, model, gpu, num_cls):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()

    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(log_dir=logdir)
    net = get_model(model, num_cls=num_cls, finetune=True)

    net.cuda()
    transform = []
    target_transform = []
    datasets = [get_dataset(name, os.path.join(datadir,name))
                for name in dataset]

    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov = True, weight_decay=1e-06)
    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2,
                                           collate_fn=collate_fn,
                                           pin_memory=True)
               for dataset in datasets]

    if use_validation:
        imgext = '.png'
        valid_data = get_validation(os.path.join(datadir,dataset[0]), crop_size, imgext)
    validfreq = 200
    showlossfreq = 500
    savefre = 500
    best_score = 10000.0
    count_ = 0
    tolerance = 100
    iteration = 0
    losses = deque(maxlen=10)
    params = dict()
    params['scale'] = 5.0
    params['alpha'] = 5.0
    steps, vals = [], []
    valid_steps, valid_vals = [], []
    for im, label in roundrobin_infinite(*loaders):
        net.train()
        opt.zero_grad()

        im_v = make_variable(im, requires_grad=False)
        label_scale, label_mask = get_weight_mask(label, params)
        label_v = make_variable(label_scale, requires_grad=False)
        label_mask_v = make_variable(label_mask, requires_grad=False)

        preds = net(im_v)
        loss = weighted_loss(label_v, preds, label_mask_v)
        loss_val = loss.data.cpu().numpy().mean()

        assert not np.isnan(loss_val) ,"nan error"
        steps.append(iteration)
        vals.append(loss_val)

        if np.mod(iteration, savefre) == 0:
            torch.save(net.state_dict(), '{}.pth'.format(output))
            print('Save weights to: ', '{}.pth'.format(output))
        if iteration == 0:
            torch.save(net.state_dict(), '{}-iter{}.pth'.format(output, iteration))

        if use_validation and iteration % validfreq == 0:
            valid_pred = net.predict(valid_data['image'], batch_size = batch_size)
            valid_label_scale, valid_label_mask = get_weight_mask(valid_data['label'], params)
            valid_loss = weighted_loss(to_device(valid_label_scale, net.device_id, var =False),\
                            to_device(valid_pred, net.device_id, var =False),\
                            to_device(valid_label_mask, net.device_id, var=False))
            valid_loss_val = valid_loss.data.cpu().numpy().mean()
            valid_steps.append(iteration)
            valid_vals.append(valid_loss_val)
            print('\nValidation loss: {}, best_score: {}'.format(valid_loss_val, best_score))
            if valid_loss_val <=  best_score:
                best_score = valid_loss_val
                print('update to new best_score: {}'.format(best_score))
                torch.save(net.state_dict(), '{}-best.pth'.format(output))
                print('Save best weights to: ', '{}-best.pth'.format(output))
                count_ = 0
            else:
                count_ = count_ + 1
            if count_ >= tolerance:
                assert 0, 'performance not imporoved for so long'

        if (iteration % showlossfreq == 0) and (iteration != 0):
            display_loss(steps, vals, plot=None, name= dataset[0] + '-' + model)
            if use_validation:
                display_loss(valid_steps, valid_vals, plot=None, name= dataset[0] + '-' + model + '_valid')

        loss.backward()
        losses.append(loss_val)

        opt.step()

        # log results
        if iteration % 100 == 0:
            logging.info('Iteration {}:\t{}'
                            .format(iteration, np.mean(losses)))
            writer.add_scalar('loss', np.mean(losses), iteration)
        iteration += 1
        if step is not None and iteration % step == 0:
            logging.info('Decreasing learning rate by 0.1.')
            step_lr(optimizer, 0.1)
        if iteration % snapshot == 0:
            torch.save(net.state_dict(),
                        '{}-iter{}.pth'.format(output, iteration))
        if iteration >= iterations:
            logging.info('Optimization complete.')
            break

if __name__ == '__main__':
    main()
