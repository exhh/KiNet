import os, sys
import time
from tqdm import *

import click
import numpy as np
from  skimage.feature import peak_local_max
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn

from nureg.data.data_loader import dataset_obj
from nureg.data.data_loader import get_fcn_dataset
from nureg.models.models import get_model
from nureg.models.models import models
from nureg.util import to_tensor_raw
from nureg.tools.analysis_util import get_seed_name, printCoordsClass

import scipy.io as sio
import copy

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='NET',
              type=click.Choice(dataset_obj.keys()))
@click.option('--datadir', default='',
        type=click.Path(exists=True))
@click.option('--eval_result_folder', default='experiments',
        type=click.Path(exists=True))
@click.option('--model', default='frcn', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=1)
@click.option('--discrim_feat/--discrim_score', default=False)
def main(path, dataset, datadir, eval_result_folder, model, gpu, num_cls, discrim_feat):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if os.path.isfile(path):
        print('Evaluate model: ', path)
    else:
        print('No trained model!')
    net = get_model(model,num_cls=num_cls)
    weights_dict = torch.load(path, map_location=lambda storage, loc: storage)
    net.load_state_dict(weights_dict)
    net.eval()

    data_split_pool = ['test']
    for data_split in data_split_pool:
        print('Evaluate ' + data_split)
        ds = get_fcn_dataset(dataset, os.path.join(datadir,dataset), split=data_split)
        loader = torch.utils.data.DataLoader(ds, num_workers=8)

        if len(loader) == 0:
            print('Empty data loader')
            return
        else:
            train_model_path = path.split('/',2)[1] + '_' + path.rsplit('/',1)[1].split('.')[0]
            savefolder = os.path.join(eval_result_folder, data_split, dataset, train_model_path)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)

        resultsDict = {}
        votingmap_name = 'votingmap'
        voting_time_name = 'prediction_time'
        detectmap_name = 'detectmap'
        classmap_name = 'classmap'
        threshold_pool = np.arange(0.0, 1.01, 0.05)
        local_min_pool = [5]
        gd_radius = 16

        iterations = tqdm(enumerate(loader))
        for im_i, (im, _, im_name) in iterations:
            resultDictPath_mat = os.path.join(savefolder, im_name[0] + '.mat')

            im = Variable(im.cuda())
            votingStarting_time = time.time()
            VotingMap = net.predict(im)
            if isinstance(VotingMap, list):
                VotingMap = np.squeeze(VotingMap[0])
            else:
                VotingMap = np.squeeze(VotingMap)
            votingEnding_time = time.time()
            print("prediction time: ", votingEnding_time - votingStarting_time)

            # Check if VotingMap is num_cls x H x W
            assert len(VotingMap.shape) == 3, "The shape of VotingMap is not correct"
            DetectMap = np.amax(VotingMap, axis=0)
            ClassMap = np.argmax(VotingMap, axis=0) + 1 # class labels begin with 1
            resultsDict[votingmap_name] = np.copy(VotingMap)
            resultsDict[detectmap_name] = np.copy(DetectMap)
            resultsDict[classmap_name] = np.copy(ClassMap)
            resultsDict[voting_time_name] = votingEnding_time - votingStarting_time

            for threshhold in threshold_pool:
                DetectMap_copy = copy.deepcopy(DetectMap)
                DetectMap_copy[DetectMap_copy < threshhold*np.max(DetectMap_copy[:])] = 0
                for min_len in local_min_pool:
                    localseedname = get_seed_name(threshhold, min_len)
                    localseedlabel = get_seed_name(threshhold, min_len) + '_label'
                    localseedtime = get_seed_name(threshhold, min_len) + '_time'

                    thisStart = time.time()
                    coordinates = peak_local_max(DetectMap_copy, min_distance= min_len, indices = True) # N x 2
                    thisEnd = time.time()

                    if coordinates.size == 0:
                        coordinates = np.asarray([])
                        labels = np.asarray([])
                        print("Detect: Empty coordinates for img:{s} for parameter t_{thd:3.2f}_r_{rad:3.2f}".format(s=im_name[0], thd=threshhold, rad=min_len))
                    else:
                        labels = ClassMap[coordinates[:,0].astype(int), coordinates[:,1].astype(int)]

                    resultsDict[localseedname] = coordinates
                    resultsDict[localseedlabel] = labels
                    resultsDict[localseedtime] = thisEnd - thisStart +  resultsDict[voting_time_name]

            sio.savemat(resultDictPath_mat, resultsDict)

        # overlay predictions on images
        imgfolder = os.path.join(datadir, dataset, 'images', data_split)
        resultfolder = savefolder
        printCoordsClass(savefolder, resultfolder, imgfolder, ['.png'], threshhold=threshold_pool[10], min_len=local_min_pool[0])

if __name__ == '__main__':
    main()
