import cv2
import numpy as np
import os
import shutil
from scipy.io import loadmat

class myobj(object):
    pass

def get_seed_name(threshhold, min_len):
    name  =('t_'   + '{:01.02f}'.format(threshhold) \
             + '_r_'+  '{:02.02f}'.format(min_len)).replace('.','_')
    return name

def getfilelist(Imagefolder, inputext):
    '''inputext: ['.json'] '''
    if type(inputext) is not list:
        inputext = [inputext]
    filelist = []
    filenames = []
    for f in os.listdir(Imagefolder):
        if os.path.splitext(f)[1] in inputext and os.path.isfile(os.path.join(Imagefolder,f)):
               filelist.append(os.path.join(Imagefolder,f))
               filenames.append(os.path.splitext(os.path.basename(f))[0])
    return filelist, filenames

def printImage(Img=None, coordinates=None, labels=None, savepath=None, **kwargs):
    '''
    print the coordinates onto the Image
    Img should be (row, col,channel)
    coordinates: should be (n,2) with (row, col) order
    return overlaiedRes
    '''
    param = myobj()
    param.linewidth = 7
    param.hlinewidth = (param.linewidth - 1) // 2
    param.color = [[0,255,255],[0,0,255],[0,255,0],[255,0,0]]
    param.alpha = 1
    for key in kwargs:
        setattr(param, key, kwargs[key])

    img_shape = Img.shape
    if coordinates.size != 0:
        coor_shape = coordinates.shape
        for idx in range(coor_shape[0]):
            x1 = int(round(coordinates[idx,0])) - param.hlinewidth
            x2 = int(round(coordinates[idx,0])) + param.hlinewidth + 1
            y1 = int(round(coordinates[idx,1])) - param.hlinewidth
            y2 = int(round(coordinates[idx,1]))+ param.hlinewidth + 1
            if x1 >= 0 and x2 <= img_shape[0] and y1 >= 0 and y2 <= img_shape[1]:
                label_idx = labels[idx] - 1
                color_b = np.full((param.linewidth,param.linewidth),param.color[label_idx][0])
                color_g = np.full((param.linewidth,param.linewidth),param.color[label_idx][1])
                color_r = np.full((param.linewidth,param.linewidth),param.color[label_idx][2])
                color_dot = cv2.merge((color_b, color_g, color_r))
                Img[x1:x2, y1:y2] = color_dot

    if savepath:
       cv2.imwrite(savepath, Img)
    return Img

def printCoordsClass(savefolder, resultfolder, imgdir, imgext, threshhold = 0.50, min_len = 5):
    imglist_, imagenamelist_ = getfilelist(imgdir, imgext)
    valid_ind = range(0, len(imglist_))

    imglist = [imglist_[i] for i in valid_ind]
    imagenamelist = [imagenamelist_[i] for i in valid_ind]

    ol_folder = os.path.join(savefolder, get_seed_name(threshhold, min_len))
    if os.path.exists(ol_folder):
       shutil.rmtree(ol_folder)
    os.makedirs(ol_folder)
    for imgindx in range(0,len(imglist)):
        print('overlay image {ind}'.format(ind = imgindx + 1))
        assert os.path.isfile(imglist[imgindx]), 'image does not exist!'
        thisimg = cv2.imread(imglist[imgindx])
        imgname = imagenamelist[imgindx]
        savepath = os.path.join(ol_folder, imgname + '_ol.png' )
        resultDictPath = os.path.join(resultfolder, imgname +  '.mat')
        print(resultDictPath)
        if os.path.isfile(resultDictPath):
           resultsDict = loadmat(resultDictPath)
        localseedname = get_seed_name(threshhold, min_len)
        coordinates = resultsDict[localseedname]
        localseedlabel = localseedname + '_label'
        labels = resultsDict[localseedlabel]
        if labels.shape[1] == 1:
            labels = labels[0,:]
        else:
            labels = np.squeeze(labels)
        printImage(Img=thisimg, coordinates=coordinates, labels=labels, savepath=savepath)
