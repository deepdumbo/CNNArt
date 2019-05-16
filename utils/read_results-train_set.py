#!/usr/bin/env python
# coding: utf-8

# In[2]:


# filepath = '/home/so2liu/Documents/Gadgetron_CNNArt/MRPhysics/NAKO_IQA/Q1/dicom_sorted/3D_GRE_TRA_fb_deep_F_COMPOSED_0043/img.npy'
filepath = '/home/so2liu/Documents/Gadgetron_CNNArt/MRPhysics/newProtocol/01_ab/dicom_sorted/t2_tse_tra_fs_navi_Leber_0006/img.npy'
import numpy as np
import time
from matplotlib import pyplot as plt
import h5py
import numpy as np
origin_img = np.load(filepath)
print('origin_img.shape = ', origin_img.shape)


# In[3]:


# read IMA files in a folder and generate a npy file
# folder_path = '/home/so2liu/Documents/Gadgetron_CNNArt/MRPhysics/newProtocol/'
# import os
# import pydicom
# import numpy as np
# img_array = np.zeros(())
# for subfolder_Q in os.listdir(folder_path):
#     subpath_Q = os.path.join(folder_path, subfolder_Q)
#     if os.path.isfile(subpath_Q):
#         continue
#     subpath_di = os.path.join(subpath_Q, 'dicom_sorted')
#     for subfolder_3D in os.listdir(subpath_di):
#         subpath_ima = os.path.join(subpath_di, subfolder_3D)
#         first_ima_path = os.path.join(subpath_ima, os.listdir(subpath_ima)[0])
#         single_size = pydicom.read_file(first_ima_path).pixel_array.shape
#         img_array = np.zeros((len(os.listdir(subpath_ima)),)+single_size)
#         print(img_array.shape)
#         for index, ima_file in enumerate(os.listdir(subpath_ima)):
#             if '.IMA' not in ima_file:
#                 continue
#             img_array[index, :, :] = pydicom.read_file(os.path.join(subpath_ima, ima_file), force=True).pixel_array
#         print(img_array.shape)
#         np.save(os.path.join(subpath_ima, 'img.npy'), img_array)
        


# ## Patching the image

# In[4]:


# a pad-calculation funciton, in z,y, x constellation
def pad_size_calculate(img_shape, not_overlap_shape, overlap_shape):
    z, y, x = tuple(img_shape)
    nonover_z, nonover_y, nonover_x = tuple(not_overlap_shape)
    img_shape, not_overlap_shape, overlap_shape = np.array(img_shape), np.array(not_overlap_shape), np.array(overlap_shape)
        
    number_patches = np.ceil(np.divide(img_shape, not_overlap_shape)).astype(np.int32)
    padded_img_shape = (np.multiply(number_patches, not_overlap_shape)+overlap_shape).astype(np.int32)
    half_pad_shape = (padded_img_shape-img_shape).astype(np.float32)/2
    x_pad = (np.ceil(half_pad_shape[2]).astype(np.int32), np.floor(half_pad_shape[2]).astype(np.int32))
    y_pad = (np.ceil(half_pad_shape[1]).astype(np.int32), np.floor(half_pad_shape[1]).astype(np.int32))
    z_pad = (0, (half_pad_shape[0]*2).astype(np.int32))
    return z_pad, y_pad, x_pad, padded_img_shape


# In[5]:


# test part for pad_size_calculate function
pad_size_calculate(origin_img.shape, [16, 77, 77], [0, 51, 51])


# In[6]:


# use z y x constellation
def f3DPatching(img_origial, patch_size, overlap_rate):
    patch_size.reverse()  # as z y x constellation
    patch_size = np.array(patch_size, dtype=np.int32)
    try:
        assert len(img_origial.shape) == 3 and len(patch_size) == 3 and overlap_rate < 1
    except Exception as e:
        print('Only support 3D image as input. The input is', img_origial.shape, patch_size, overlap_rate)
        print('Reason:', e)     
    origin_size = img_origial.shape
    overlap_pixel_yx = np.floor(overlap_rate*patch_size[1:]).astype(np.int32)
    not_overlap_pixel_yx = np.ndarray.astype(patch_size[1:]-overlap_pixel_yx, np.int32)

    # padding    
    z_pad, y_pad, x_pad, _ = pad_size_calculate(origin_size, 
                                             not_overlap_shape=(patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1]), 
                                             overlap_shape=(0, overlap_pixel_yx[0], overlap_pixel_yx[1]))
    padded_img = np.pad(img_origial, (z_pad, y_pad, x_pad), 'constant')
    assert not np.any(np.mod(padded_img.shape[1:], not_overlap_pixel_yx)-overlap_pixel_yx)  # ensure all zeros
    assert not np.any(np.mod(padded_img.shape[0], patch_size[0]))  # ensure all zeros    
        
    # patching 
    def fast_3D_strides(img, patch_shape, stepsize_tuple): 
        z, y, x = img.shape
        step_z, step_y, step_x = stepsize_tuple
        sz, sy, sx = img.strides
        
        patch_frame_shape = np.divide(np.array(img.shape)-(np.array(patch_shape)-np.array(stepsize_tuple)), np.array(stepsize_tuple))
        print('patch_frame_shape =', patch_frame_shape)
        patch_frame_shape = tuple(patch_frame_shape.astype(int))  # big patch struction
        result_shape = patch_frame_shape+tuple(patch_shape)
        print('result_shape =', result_shape)
        result_strides = (sx*step_z*x*y, sx*x*step_y, sx*step_x, sx*x*y, sx*x, sx)    
        return np.lib.stride_tricks.as_strided(img, result_shape, result_strides)

  
    patched_img = fast_3D_strides(padded_img, patch_shape=patch_size, 
                                  stepsize_tuple=(patch_size[0], 
                                                  not_overlap_pixel_yx[0], 
                                                  not_overlap_pixel_yx[1]))  # axis2 is full not-overlap
    patched_img = np.reshape(patched_img, (-1, )+tuple(patch_size))

    return np.swapaxes(patched_img, 1, -1)  # return (n, x, y, z) constellation


# In[7]:


# test part for patching function
patched_img = f3DPatching(origin_img, patch_size=[128, 128, 16], overlap_rate=0.4)
print('patched_img.shape =', patched_img.shape)

hight_index = 5

plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(origin_img[hight_index, :, :])
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
a = np.swapaxes(patched_img, 1, 2)  # yxz constellation is more suitable for comparision

big_z = hight_index//16
n_z = big_z*20
small_z = hight_index % 16
for i in range(n_z, n_z+20):
    plt.subplot(4, 5, i+1-n_z)
    plt.imshow(a[i, :, :, small_z])


# ## Unpatching subfunction
# which returns results as the size or original image. 

# In[8]:


def f3DUnpatching(patched_img, origin_shape, patch_size, overlap_rate):
#     input: 4d imag with (n, x, y, z) as constellation,  1d patch_size (3 elements), a float overlap_rate
#     output: non-padded, non-seperated, non-overlapped 3d img with size of patch_size in zyx constellation
    assert len(patched_img.shape) == 4 and isinstance(patch_size, list)
    patched_img = patched_img.copy()
    # change everything into z, y, x  constellation
    patched_img = np.swapaxes(patched_img, 1, -1)  # switch x and z
    patch_size.reverse()
    patch_size = np.array(patch_size)

    # calculate pad size
    overlap_pixel_yx = np.floor(overlap_rate*patch_size[1:]).astype(np.int32)
    not_overlap_pixel_yx = (patch_size[1:]-overlap_pixel_yx).astype(np.int32)
    stepsize_tuple = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    z_pad, y_pad, x_pad, padded_img_shape = pad_size_calculate(origin_shape, 
                                             not_overlap_shape=(patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1]), 
                                             overlap_shape=(0, overlap_pixel_yx[0], overlap_pixel_yx[1]))    
    rest_over_tuple = (0, overlap_pixel_yx[0], overlap_pixel_yx[1])
    patch_frame_shape = tuple(np.divide(np.array(padded_img_shape)-np.array(rest_over_tuple), np.array(stepsize_tuple)).astype(int))  # big patch struction

    non_overlap_patch_shape = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    non_overlap_patched_img = np.zeros((patched_img.shape[0], )+non_overlap_patch_shape)
    
    # padded, seperated, overlapped img => padded, seperated, non-overlapped img
    for n in range(patched_img.shape[0]):
        non_overlap_patched_img[n, :, :, :] = patched_img[n, 0:patch_size[0], 0: not_overlap_pixel_yx[0], 0: not_overlap_pixel_yx[1]]
    
    framed_img = np.reshape(non_overlap_patched_img, patch_frame_shape+non_overlap_patch_shape)
    
    # padded, seperated, non-overlapped img => padded, non-seperated, non-overlapped img
    result = np.ones(padded_img_shape)*255
    overlap_framed_img = np.reshape(patched_img, patch_frame_shape+tuple(patch_size))
    for fz in range(patch_frame_shape[0]):
        for fy in range(patch_frame_shape[1]-1):  # -1 for ignore the right and bottom margin
            for fx in range(patch_frame_shape[2]-1):
                result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                       fy*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                       fx*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = framed_img[fz, fy, fx, :, :, :]
                # add the margin at right and bottom of the big img
                if fy == patch_frame_shape[1]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy+1)*non_overlap_patch_shape[1]:,
                           (fx)*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = overlap_framed_img[fz, fy+1, fx, :, :, :not_overlap_pixel_yx[1]]
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           -patch_size[1]:,
                           -patch_size[2]:] = overlap_framed_img[fz, fy+1, fx+1, :, :, :]
                    
                if fx == patch_frame_shape[2]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy)*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                           (fx+1)*non_overlap_patch_shape[2]:] = overlap_framed_img[fz, fy, fx+1, :, :not_overlap_pixel_yx[0], :]
    
    # padded, non-seperated, non-overlapped img => non-padded, non-seperated, non-overlapped img
    result = result[z_pad[0]:-z_pad[1], y_pad[0]:-y_pad[1], x_pad[0]:-x_pad[1]]
    assert result.shape == tuple(origin_shape)    
    return result/result.max() # normalization


# In[9]:


def f3DUnpatching(patched_img, origin_shape, patch_size, overlap_rate, overlap_method='average'):
#     input: 4d imag with (n, x, y, z) as constellation,  1d patch_size (3 elements), a float overlap_rate, method dealing with overlap => ('average', 'cutoff')
#     output: non-padded, non-seperated, non-overlapped 3d img with size of patch_size in zyx constellation
    patched_img = patched_img.copy()  # otherwise the patched_img in main() will be changed
    assert len(patched_img.shape) == 4 and isinstance(patch_size, list)
    assert overlap_method in ('average', 'cutoff')
    
    # change everything into z, y, x  constellation
    patched_img = np.swapaxes(patched_img, 1, -1)  # switch x and z
    patch_size.reverse()
    patch_size = np.array(patch_size)

    # calculate pad size
    overlap_pixel_yx = np.floor(overlap_rate*patch_size[1:]).astype(np.int32)
    not_overlap_pixel_yx = (patch_size[1:]-overlap_pixel_yx).astype(np.int32)
    stepsize_tuple = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    z_pad, y_pad, x_pad, padded_img_shape = pad_size_calculate(origin_shape, 
                                             not_overlap_shape=(patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1]), 
                                             overlap_shape=(0, overlap_pixel_yx[0], overlap_pixel_yx[1]))    
    rest_over_tuple = (0, overlap_pixel_yx[0], overlap_pixel_yx[1])
    patch_frame_shape = tuple(np.divide(np.array(padded_img_shape)-np.array(rest_over_tuple), np.array(stepsize_tuple)).astype(int))  # big patch struction

    non_overlap_patch_shape = (patch_size[0], not_overlap_pixel_yx[0], not_overlap_pixel_yx[1])
    non_overlap_patched_img = np.zeros((patched_img.shape[0], )+non_overlap_patch_shape)
    
    
    if overlap_method == 'average':
        for n in range(patched_img.shape[0]):
            if n%patch_frame_shape[2] == 0:
                continue
            this_img_start = patched_img[n, :, :, :overlap_pixel_yx[1]]
            last_img_end = patched_img[n-1, :, :, -overlap_pixel_yx[1]:]
#             print(n, this_img_start.shape, last_img_end.shape)
            patched_img[n, :, :, :overlap_pixel_yx[1]] = (this_img_start+last_img_end)/2
        for n in range(patched_img.shape[0]):
            if n%patch_frame_shape[1] == 0:
                continue
            this_img_start = patched_img[n, :, :overlap_pixel_yx[0], :]
            last_img_end = patched_img[n-patch_frame_shape[2], :, -overlap_pixel_yx[0]:, :]
#             print(n, this_img_start.shape, last_img_end.shape)
            patched_img[n, :, :overlap_pixel_yx[0], :] = (this_img_start+last_img_end)/2
        
        
    # padded, seperated, overlapped img => padded, seperated, non-overlapped img
    for n in range(patched_img.shape[0]):
        non_overlap_patched_img[n, :, :, :] = patched_img[n, 0:patch_size[0], 0: not_overlap_pixel_yx[0], 0: not_overlap_pixel_yx[1]]
    framed_img = np.reshape(non_overlap_patched_img, patch_frame_shape+non_overlap_patch_shape)
            
    # padded, seperated, non-overlapped img => padded, non-seperated, non-overlapped img
    result = np.ones(padded_img_shape)*255
    overlap_framed_img = np.reshape(patched_img, patch_frame_shape+tuple(patch_size))
    for fz in range(patch_frame_shape[0]):
        for fy in range(patch_frame_shape[1]-1):  # -1 for ignore the right and bottom margin
            for fx in range(patch_frame_shape[2]-1):
                result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                       fy*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                       fx*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = framed_img[fz, fy, fx, :, :, :]
                # add the margin at right and bottom of the big img
                if fy == patch_frame_shape[1]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy+1)*non_overlap_patch_shape[1]:,
                           (fx)*non_overlap_patch_shape[2]:(fx+1)*non_overlap_patch_shape[2]] = overlap_framed_img[fz, fy+1, fx, :, :, :not_overlap_pixel_yx[1]]
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           -patch_size[1]:,
                           -patch_size[2]:] = overlap_framed_img[fz, fy+1, fx+1, :, :, :]
                    
                if fx == patch_frame_shape[2]-2:
                    result[fz*non_overlap_patch_shape[0]:(fz+1)*non_overlap_patch_shape[0], 
                           (fy)*non_overlap_patch_shape[1]:(fy+1)*non_overlap_patch_shape[1],
                           (fx+1)*non_overlap_patch_shape[2]:] = overlap_framed_img[fz, fy, fx+1, :, :not_overlap_pixel_yx[0], :]
    
    # padded, non-seperated, non-overlapped img => non-padded, non-seperated, non-overlapped img
    result = result[z_pad[0]:-z_pad[1], y_pad[0]:-y_pad[1], x_pad[0]:-x_pad[1]]
    assert result.shape == tuple(origin_shape)    
    return result/result.max() # normalization


# In[10]:


# test for unpatching
predict_img = f3DUnpatching(patched_img, origin_img.shape, [128, 128, 16], 0.4)
print('predict_img.shape =', predict_img.shape)
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(origin_img[5, :, :])        
plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
plt.imshow(predict_img[5, :, :])


# ## Make prediction using model.

# In[10]:


# imports
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os.path                      # operating system
import argparse
import yaml
import gc
import keras
import pickle
import joblib  # pickle will cause memory error for the large numpy file
import time

def dice_coef(y_true, y_pred, epsilon=1e-5):
    dice_numerator = 2.0 * K.sum(y_true*y_pred, axis=[1,2,3,4])
    dice_denominator = K.sum(K.square(y_true), axis=[1,2,3,4]) + K.sum(K.square(y_pred), axis=[1,2,3,4])
    dice_score = dice_numerator / (dice_denominator + epsilon)
    return K.mean(dice_score, axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

if os.getcwd()+'/CNNArt' not in sys.path:
    sys.path.append(os.getcwd()+'/CNNArt')
# utils
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import utils.scaling as scaling

from networks.FullyConvolutionalNetworks.motion.VResFCN_3D_Upsampling_final_Motion_Binary import fPredict 

# VAE correction network
# import correction.main_correction as correction  #  can't import, don't know why
# multi-scale
from utils.calculateInputOfPath2 import fcalculateInputOfPath2
from networks.multiscale.runMS import frunCNN_MS

with open('CNNArt/config/param_Gadgetron.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# prediction
sTrainingMethod = cfg['sTrainingMethod']
patchSize = cfg['patchSize']
sNetworktype = cfg['network'].split("_")
sPredictModel = cfg['sPredictModel']
lTrain = cfg['lTrain']
sOutPath = 'result'

# X_test = np.expand_dims(X_test, axis=-1)
ModelPath = '/home/so2liu/Documents/Gadgetron_CNNArt/cnnart_trainednets/motion/FCN'
sFilename = 'FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2019-03-28_18-46'
with open(ModelPath + os.sep + sFilename + '.json', 'r') as fp:
    model = keras.models.model_from_json(fp.read())

model.load_weights(ModelPath+os.sep+sFilename+'_weights.h5')

prob_pre = model.predict(np.expand_dims(patched_img, axis=-1), batch_size=2, verbose=1)

prob_pre_filename = 'prob_pre'+str(int(time.time()))+'.joblib'
with open(prob_pre_filename, 'wb') as f:
    joblib.dump(prob_pre, f)
print('Prediction data is saved as', prob_pre_filename)


# In[34]:


if os.getcwd()+'/CNNArt' not in sys.path:
    sys.path.append(os.getcwd()+'/CNNArt')
# imports
import sys
import numpy as np                  # for algebraic operations, matrices
import h5py
import scipy.io as sio              # I/O
import os.path                      # operating system
import argparse

# utils
from DatabaseInfo import DatabaseInfo
import utils.DataPreprocessing as datapre
import utils.Training_Test_Split as ttsplit
import utils.scaling as scaling

# networks
from networks.motion.CNN2D import *
from networks.motion.CNN3D import *
from networks.motion.MNetArt import *
from networks.motion.VNetArt import *
from networks.multiclass.CNN2D.DenseResNet import *
from networks.multiclass.CNN2D.InceptionNet import *
from correction.networks.motion import *
from networks.FullyConvolutionalNetworks.motion import *

# VAE correction network
import correction.main_correction as correction

# multi-scale
from utils.calculateInputOfPath2 import fcalculateInputOfPath2
from networks.multiscale.runMS import frunCNN_MS

#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim

def fRunCNN(dData, sModelIn, lTrain, sParaOptim, sOutPath, iBatchSize, iLearningRate, iEpochs, CV_Patient=0):
    """CNN Models"""
    # check model
    if 'motion' in sModelIn:
        if 'CNN2D' in sModelIn:
            sModel = 'networks.motion.CNN2D.' + sModelIn
        elif 'motion_CNN3D' in sModelIn:
            sModel = 'networks.motion.CNN3D.' + sModelIn
        elif 'motion_MNetArt' in sModelIn:
            sModel = 'networks.motion.MNetArt.' + sModelIn
        elif 'motion_VNetArt' in sModelIn:
            sModel = 'networks.motion.VNetArt.' + sModelIn
    elif 'FCN' in sModelIn:
        sModel = 'networks.FullyConvolutionalNetworks.motion.' + sModelIn # TODO: may require to adapt patching and data augmentation from GUI/PyQt/DLart/dlart.py
    elif 'multi' in sModelIn:
        if 'multi_DenseResNet' in sModelIn:
            sModel = 'networks.multiclass.DenseResNet.' + sModelIn
        elif 'multi_InceptionNet' in sModelIn:
            sModel = 'networks.multiclass.InceptionNet.' + sModelIn
    else:
        sys.exit("Model is not supported")

    # dynamic loading of corresponding model
    cnnModel = __import__(sModel, globals(), locals(), ['createModel', 'fTrain', 'fPredict'], 0)  # dynamic module loading with specified functions and with absolute importing (level=0) -> work in both Python2 and Python3

    # train (w/ or w/o optimization) and predicting
    if lTrain:  # training
        if sParaOptim == 'hyperas':  # hyperas parameter optimization
            best_run, best_model = optim.minimize(model=cnnModel.fHyperasTrain,
                                                  data=fLoadDataForOptim(args.inPath[0]),
                                                  algo=tpe.suggest,
                                                  max_evals=5,
                                                  trials=Trials())
            X_train, y_train, X_test, y_test, patchSize = fLoadDataForOptim(args.inPath[0])
            score_test, acc_test = best_model.evaluate(X_test, y_test)
            prob_test = best_model.predict(X_test, best_run['batch_size'], 0)

            _, sPath = os.path.splitdrive(sOutPath)
            sPath, sFilename = os.path.split(sPath)
            sFilename, sExt = os.path.splitext(sFilename)
            model_name = sPath + '/' + sFilename + str(patchSize[0, 0]) + str(patchSize[0, 1]) + '_best'
            weight_name = model_name + '_weights.h5'
            model_json = model_name + '_json'
            model_all = model_name + '_model.h5'
            json_string = best_model.to_json()
            open(model_json, 'w').write(json_string)
            # wei = best_model.get_weights()
            best_model.save_weights(weight_name)
            # best_model.save(model_all)

            result = best_run['result']
            # acc = result.history['acc']
            loss = result.history['loss']
            val_acc = result.history['val_acc']
            val_loss = result.history['val_loss']
            sio.savemat(model_name, {'model_settings': model_json,
                                     'model': model_all,
                                     'weights': weight_name,
                                     'acc': -best_run['loss'],
                                     'loss': loss,
                                     'val_acc': val_acc,
                                     'val_loss': val_loss,
                                     'score_test': score_test,
                                     'acc_test': acc_test,
                                     'prob_test': prob_test})

        elif sParaOptim == 'grid':  # grid search << backward compatibility
            cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient)

        else:# no optimization or grid search (if batchSize|learningRate are arrays)
            cnnModel.fTrain(dData['X_train'], dData['y_train'], dData['X_test'], dData['y_test'], sOutPath, dData['patchSize'], iBatchSize, iLearningRate, iEpochs, CV_Patient=CV_Patient)

    else:  # predicting
        cnnModel.fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, patchSize=dData['patchSize'], batchSize=iBatchSize[0], patchOverlap=dData['patchOverlap'], actualSize=dData['actualSize'], iClass=dData['iClass'])


# In[35]:


import yaml
with open('CNNArt/config/param_Gadgetron.yml', 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# prediction
sTrainingMethod = cfg['sTrainingMethod']
patchSize = cfg['patchSize']
sNetworktype = cfg['network'].split("_")
sPredictModel = cfg['sPredictModel']
lTrain = cfg['lTrain']
sOutPath = 'result'

dData = patched_img
sModelIn = '/home/so2liu/Documents/Gadgetron_CNNArt/cnnart_trainednets/motion/FCN'
sParaOptim = None
iBatchSize = 2
iLearningRate = None
iEpochs = cfg['epochs']
cnnModel.fPredict(dData['X_test'], dData['y_test'], dData['model_name'], sOutPath, patchSize=dData['patchSize'], batchSize=iBatchSize[0], patchOverlap=dData['patchOverlap'], actualSize=dData['actualSize'], iClass=dData['iClass'])


# In[36]:


from keras.models import model_from_json

def fPredict(X_test, y=None, Y_segMasks_test=None, sModelPath=None, sFilename=None, sOutPath=None, batch_size=64):
    """Takes an already trained model and computes the loss and Accuracy over the samples X with their Labels y
        Input:
            X: Samples to predict on. The shape of X should fit to the input shape of the model
            y: Labels for the Samples. Number of Samples should be equal to the number of samples in X
            sModelPath: (String) full path to a trained keras model. It should be *_json.txt file. there has to be a corresponding *_weights.h5 file in the same directory!
            sOutPath: (String) full path for the Output. It is a *.mat file with the computed loss and accuracy stored.
                        The Output file has the Path 'sOutPath'+ the filename of sModelPath without the '_json.txt' added the suffix '_pred.mat'
            batchSize: Batchsize, number of samples that are processed at once"""

    X_test = np.expand_dims(X_test, axis=-1)
#     Y_segMasks_test_foreground = np.expand_dims(Y_segMasks_test, axis=-1)
#     Y_segMasks_test_background = np.ones(Y_segMasks_test_foreground.shape) - Y_segMasks_test_foreground
#     Y_segMasks_test = np.concatenate((Y_segMasks_test_background, Y_segMasks_test_foreground), axis=-1)

#     _, sPath = os.path.splitdrive(sModelPath)
#     sPath, sFilename = os.path.split(sPath)
#     sFilename, sExt = os.path.splitext(sFilename)

#     listdir = os.listdir(sModelPath)

    #sModelPath = sModelPath.replace("_json.txt", "")
    #weight_name = sModelPath + '_weights.h5'
    #model_json = sModelPath + '_json.txt'
    #model_all = sModelPath + '_model.h5'

    # load weights and model (new way)
    with open(sModelPath + os.sep + sFilename + '.json', 'r') as fp:
        model_string = fp.read()

    model = model_from_json(model_string)

#     model.summary()

    model.compile(loss=dice_coef_loss, optimizer=keras.optimizers.Adam(), metrics=[dice_coef])
    model.load_weights(sModelPath+ os.sep + sFilename+'_weights.h5')

    score_test, acc_test = model.evaluate(X_test, Y_segMasks_test, batch_size=2)
    print('loss' + str(score_test) + '   acc:' + str(acc_test))

    prob_pre = model.predict(X_test, batch_size=batch_size, verbose=1)

    predictions = {'prob_pre': prob_pre, 'score_test': score_test, 'acc_test': acc_test}

    return predictions


# In[37]:


ModelPath = '/home/so2liu/Documents/Gadgetron_CNNArt/cnnart_trainednets/motion/FCN'
sFilename = 'FCN 3D-VResFCN-Upsampling final Motion Binary_3D_128x128x16_2019-03-28_18-46'

predictions = fPredict(X_test=patched_img, sModelPath=ModelPath, sFilename=sFilename, batch_size=2)


# In[11]:


plt.imshow(prob_pre[9, :, :, 5, 0], cmap='gray')
del prob_pre


# ## Read saved prediction

# In[1]:


import joblib
# prob_pre_filename = 'prob_pre1556791615.joblib'
with open(prob_pre_filename, 'rb') as f:
    predictions = joblib.load(f)
print('predicted result predictions.shape = ', predictions.shape)


# In[13]:


# not good
# import pickle
# with open('prob_pre1556181935', 'rb') as f:
#     predictions = pickle.load(f)
# print('predicted result predictions.shape = ', predictions.shape)


# ## Unpatch the prediction and imshow

# In[14]:


predictions[:, :, :, :, 0].shape


# In[15]:


patched_img.shape


# In[17]:


predict_img = f3DUnpatching(patched_img, origin_img.shape, [128, 128, 16], 0.4)
predict_result = f3DUnpatching(predictions[:, :, :, :, 0], origin_img.shape, [128, 128, 16], 0.4)
print('predictions.shape =', predictions.shape)

transparency = 0.1
hight_index = 5
this_img = origin_img[hight_index, :, :].copy()
this_img = this_img/this_img.max()
this_prediction = predict_result[hight_index, :, :].copy()
this_prediction = this_prediction/this_prediction.max()*3
plt_img = plt.figure(num=None, figsize=(15, 15), dpi=150, facecolor='w', edgecolor='k')
plt.subplot(3, 2, 1)
plt.gca().set_title('Original image with z='+str(hight_index))
plt.imshow(this_img, cmap='gray', vmax=this_img.max()*0.3)        
plt.subplot(3, 2, 2)
plt.gca().set_title('Unpatched image (showing unpatched subfunction works)')
plt.imshow(predict_img[hight_index, :, :], cmap='gray', vmax=predict_img[hight_index, :, :].max()*0.3)
plt.subplot(3, 2, 3)
plt.gca().set_title('Predictions for z='+str(hight_index))
plt.imshow(this_prediction, cmap='gray', vmax=this_prediction.max()*0.3)
plt.subplot(3, 2, 4)
plt.gca().set_title('Combined image from original image and predictions')
this_img_prediction = np.stack((this_img,)*3, axis=-1)+np.moveaxis(np.array(
    [np.zeros(this_prediction.shape), this_prediction, np.zeros(this_prediction.shape)]), 0, -1)*transparency  # last number is transparency
plt.imshow(this_img_prediction/this_img_prediction.max()*5)  # normalization

plt.subplot(3, 2, 5)
layer_index = hight_index//16
patched_n_index = layer_index*20+6
patched_z_index = hight_index%20
this_patched_img = patched_img[patched_n_index, :, :, patched_z_index].copy()
this_patched_img = this_patched_img/this_patched_img.max()
this_patched_prediction = predictions[patched_n_index, :, :, patched_z_index, 0].copy()
this_patched_prediction = this_patched_prediction/this_patched_prediction.max()
# this_patched_prediction = np.zeros((128, 128))

this_patched_img_prediction = np.stack((this_patched_img, )*3, axis=-1)+np.moveaxis(np.array(
    [np.zeros(this_patched_prediction.shape), this_patched_prediction, np.zeros(this_patched_prediction.shape)]), 0, -1)*transparency

this_patched_img_prediction = np.swapaxes(this_patched_img_prediction, 0, 1)
plt.gca().set_title('Single patched image of this z with No. 6')
plt.imshow(this_patched_img_prediction*1)
plt.subplot(3, 2, 6)
layer_index = hight_index//16
patched_n_index = layer_index*20+8
patched_z_index = hight_index%20
this_patched_img = patched_img[patched_n_index, :, :, patched_z_index]
this_patched_img = this_patched_img/this_patched_img.max()
this_patched_prediction = predictions[patched_n_index, :, :, patched_z_index, 0]
this_patched_prediction = this_patched_prediction/this_patched_prediction.max()

this_patched_img_prediction = np.stack((this_patched_img, )*3, axis=-1)+np.moveaxis(np.array(
    [np.zeros(this_patched_prediction.shape), this_patched_prediction, np.zeros(this_patched_prediction.shape)]), 0, -1)*transparency

this_patched_img_prediction = np.swapaxes(this_patched_img_prediction, 0, 1)
plt.gca().set_title('Single patched image of this z with No. 8')
plt.imshow(this_patched_img_prediction*1)
import time
img_name = 'cnnart_'+str(int(time.time()))+'.png'
plt.savefig(img_name)
print('Images are saved in', img_name)


# In[7]:


predictions[:, :, :, :, 1]+predictions[:, :, :, :, 0]


# In[ ]:




