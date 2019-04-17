"""
@author: Sebastian Milde, Thomas Kuestner
"""
import numpy as np
import math
import random
from utils.scaling import fScaleOnePatch
import tensorflow as tf

#########################################################################################################################################
#Function: fRigidPatching                                                                                                               #
#The function fRigidPatching is responsible for splitting the dicom numpy array in patches depending on the patchSize and the           #
#patchOverlap. Besides the function creates an 1D array with the corresponding labels.                                                  #
#                                                                                                                                       #
#Input: dicom_numpy_array ---> 3D dicom array (height, width, number of slices)                                                         #
#       patchSize ---> size of patches, example: [40, 40], patchSize[0] = height, patchSize[1] = weight, height and weight can differ   #
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                      #
#       mask_numpy_array ---> 3D mask array contains information about the areas of artefacts. movement-artefact = 1, shim-artefact = 2 #
#                             noise-artefact = 3                                                                                        #
#       ratio_labeling ---> set the ratio of the number of 'Pixel-Artefacts' to the whole number of pixels of one patch                 #
#Output: dPatches ---> 3D-Numpy-Array, which contain all Patches.                                                                       #
#        dLabels ---> 1D-Numpy-Array with all corresponding labels
                                                                     #
#########################################################################################################################################

def fRigidPatching(dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling, sLabeling):
    move_artefact = False
    shim_artefact = False
    noise_artefact = False

    dOverlap = np.multiply(patchSize, patchOverlap)
    dNotOverlap = np.round(np.multiply(patchSize, (1 - patchOverlap)))
    size_zero_pad = np.array(([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1]]))
    zero_pad = np.array(([int(math.ceil(size_zero_pad[0])) - dicom_numpy_array.shape[0], int(math.ceil(size_zero_pad[1])) - dicom_numpy_array.shape[1]]))
    zero_pad_part = np.array(([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2))]))
    Img_zero_pad = np.lib.pad(dicom_numpy_array, (
    (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                              mode='constant')
    nbPatches = int(((size_zero_pad[0]-patchSize[0])/((1-patchOverlap)*patchSize[0])+1)*((size_zero_pad[1]-patchSize[1])/((1-patchOverlap)*patchSize[1])+1)*dicom_numpy_array.shape[2])
    dPatches = np.zeros((patchSize[0], patchSize[1], nbPatches), dtype=float) #dtype=np.float32
    dLabels = np.zeros((nbPatches), dtype = float) #dtype = float
    idxPatch = 0
    if sLabeling == 'volume':
        for iZ in range(0, dicom_numpy_array.shape[2], 1):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    dPatches[:,:,idxPatch] = dPatch
                    idxPatch += 1

        #print(idxPatch)
        dPatches = dPatches[:, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[2]))
    elif sLabeling == 'patch':
        Mask_zero_pad = np.lib.pad(mask_numpy_array, (
        (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                                  mode='constant')

        for iZ in range(0, dicom_numpy_array.shape[2], 1):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    dPatches[:,:,idxPatch] = dPatch

                    dPatch_mask = Mask_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ]
                    patch_number_value = patchSize[0] * patchSize[1]

                    if np.count_nonzero((dPatch_mask==1).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        move_artefact = True
                    if np.count_nonzero((dPatch_mask==2).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        shim_artefact = True
                    if np.count_nonzero((dPatch_mask==3).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        noise_artefact = True

                    label = 0

                    if move_artefact == True and shim_artefact != True and noise_artefact != True:
                        label = 1
                    elif move_artefact != True and shim_artefact == True and noise_artefact != True:
                        label = 2
                    elif move_artefact != True and shim_artefact != True and noise_artefact == True:
                        label = 3
                    elif move_artefact == True and shim_artefact == True and noise_artefact != True:
                        label = 4
                    elif move_artefact == True and shim_artefact != True and noise_artefact == True:
                        label = 5
                    elif move_artefact != True and shim_artefact == True and noise_artefact == True:
                        label = 6
                    elif move_artefact == True and shim_artefact == True and noise_artefact == True:
                        label = 7

                    dLabels[idxPatch] = label
                    idxPatch += 1

                    move_artefact = False
                    shim_artefact = False
                    noise_artefact = False

        dPatches = dPatches[:, :, 0:idxPatch]
        dLabels = dLabels[0:idxPatch]
    return dPatches, dLabels
##########################################################################################################################################
# In case of 3D patches:                                                                                                                 #
#Input: dicom_numpy_array ---> 4D dicom array (height, width, lengh, number of slices)                                                   #
#       patchSize ---> size of patches, example: [40, 40, 10], patchSize[0] = height, patchSize[1] = weight, height and weight can differ#
#       patchOverlap ---> the ratio for overlapping, example: 0.25                                                                       #
#       mask_numpy_array ---> 4D mask array contains information about the areas of artefacts. movement-artefact = 1, shim-artefact = 2  #
#                             noise-artefact = 3                                                                                         #
#       ratio_labeling ---> set the ratio of the number of 'Pixel-Artefacts' to the whole number of pixels of one patch                  #
#Output: dPatches ---> 4D-Numpy-Array, which contain all Patches.                                                                        #
#        dLabels ---> 1D-Numpy-Array with all corresponding labels                                                                       #
##########################################################################################################################################
def fRigidPatching3D(dicom_numpy_array, patchSize, patchOverlap, mask_numpy_array, ratio_labeling, sLabeling, sTrainingMethod='None'):

    move_artefact = False
    shim_artefact = False
    noise_artefact = False


    dOverlap = np.multiply(patchSize, patchOverlap)
    dNotOverlap = np.ceil(np.multiply(patchSize, (1 - patchOverlap)))
    size_zero_pad = np.array([math.ceil((dicom_numpy_array.shape[0] - dOverlap[0]) / (dNotOverlap[0])) * dNotOverlap[0] + dOverlap[
        0], math.ceil((dicom_numpy_array.shape[1] - dOverlap[1]) / (dNotOverlap[1])) * dNotOverlap[1] + dOverlap[1], math.ceil((dicom_numpy_array.shape[2] - dOverlap[2]) / (dNotOverlap[2])) * dNotOverlap[2] + dOverlap[2]])
    zero_pad = np.array([int(math.ceil(size_zero_pad[0])) - dicom_numpy_array.shape[0], int(math.ceil(size_zero_pad[1])) - dicom_numpy_array.shape[1], int(math.ceil(size_zero_pad[2])) - dicom_numpy_array.shape[2]])
    zero_pad_part = np.array([int(math.ceil(zero_pad[0] / 2)), int(math.ceil(zero_pad[1] / 2)), int(math.ceil(zero_pad[2] / 2))])

    Img_zero_pad = np.lib.pad(dicom_numpy_array, ((zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (zero_pad_part[2], zero_pad[2] - zero_pad_part[2])),
                              mode='constant')

    nbPatches = ((size_zero_pad[0]-patchSize[0])/((1-patchOverlap)*patchSize[0])+1)*((size_zero_pad[1]-patchSize[1])/((1-patchOverlap)*patchSize[1])+1)*((size_zero_pad[2]-patchSize[2])/(np.round((1-patchOverlap)*patchSize[2]))+1)
    dPatches = np.zeros((patchSize[0], patchSize[1], patchSize[2], int(nbPatches)), dtype=float)
    dLabels = np.zeros((int(nbPatches)), dtype = int) #float
    idxPatch = 0

    if sLabeling == 'volume' and sTrainingMethod == 'ScaleJittering':
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    if (iX>=int(size_zero_pad[1] - dOverlap[1]-patchSize[1])) or(iY>=int(size_zero_pad[0] - dOverlap[0]-patchSize[0])) or (iZ>=int(size_zero_pad[2] - dOverlap[2]-patchSize[2])):
                        randPatchSize = patchSize
                    else:
                        randPatchSize = np.round(np.multiply(patchSize, (np.random.rand(1) + 1))).astype(int)
                    dPatch = Img_zero_pad[iY:iY + randPatchSize[0], iX:iX + randPatchSize[1], iZ:iZ + randPatchSize[2]]
                    scaleddPatch = fScaleOnePatch(dPatch, randPatchSize, patchSize)
                    dPatches[:, :, :, idxPatch] = scaleddPatch
                    idxPatch += 1
        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[3]))

    elif sLabeling == 'volume':
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    dPatches[:,:,:,idxPatch] = dPatch
                    idxPatch += 1

        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = np.ones((dPatches.shape[3]))
    elif sLabeling == 'patch':
        Mask_zero_pad = np.lib.pad(mask_numpy_array, (
            (zero_pad_part[0], zero_pad[0] - zero_pad_part[0]), (zero_pad_part[1], zero_pad[1] - zero_pad_part[1]), (0, 0)),
                                   mode='constant')
        for iZ in range(0, int(size_zero_pad[2] - dOverlap[2]), int(dNotOverlap[2])):
            for iY in range(0, int(size_zero_pad[0] - dOverlap[0]), int(dNotOverlap[0])):
                for iX in range(0, int(size_zero_pad[1] - dOverlap[1]), int(dNotOverlap[1])):
                    dPatch = Img_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    dPatches[:,:,:,idxPatch] = dPatch

                    dPatch_mask = Mask_zero_pad[iY:iY + patchSize[0], iX:iX + patchSize[1], iZ:iZ + patchSize[2]]
                    patch_number_value = patchSize[0] * patchSize[1] * patchSize[2]

                    if np.count_nonzero((dPatch_mask==1).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        move_artefact = True
                    if np.count_nonzero((dPatch_mask==2).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        shim_artefact = True
                    if np.count_nonzero((dPatch_mask==3).astype(np.int)) > int(ratio_labeling*patch_number_value):
                        noise_artefact = True

                    label = 0

                    if move_artefact == True and shim_artefact != True and noise_artefact != True:
                        label = 1
                    elif move_artefact != True and shim_artefact == True and noise_artefact != True:
                        label = 2
                    elif move_artefact != True and shim_artefact != True and noise_artefact == True:
                        label = 3
                    elif move_artefact == True and shim_artefact == True and noise_artefact != True:
                        label = 4
                    elif move_artefact == True and shim_artefact != True and noise_artefact == True:
                        label = 5
                    elif move_artefact != True and shim_artefact == True and noise_artefact == True:
                        label = 6
                    elif move_artefact == True and shim_artefact == True and noise_artefact == True:
                        label = 7

                    dLabels[idxPatch] = label
                    idxPatch += 1

                    move_artefact = False
                    shim_artefact = False
                    noise_artefact = False
                    
        dPatches = dPatches[:, :, :, 0:idxPatch]
        dLabels = dLabels[0:idxPatch]
    return dPatches, dLabels

########################################################################################################################
# @ author : Shanqi Yang
# input : 3D image tensor
# output : lists of pathes cropped from this 3D tensor
# calls compute_patch_indices to calculate the index lists where the image will be cropped
########################################################################################################################

def get_patches(image, num_patches, image_shape = [316, 260, 320],
                patch_size=[64, 64, 64], overlap = 32, start = [0, 0, 0]):

# Order: to define where to start to choose the index, from the start side or from the end side
    order = random.choice([False, True])
    index_lists = compute_patch_indices(image_shape = image_shape,
                                        patch_size = patch_size,
                                        overlap = overlap,
                                        start = start,
                                        order = order)
    assert num_patches == len(index_lists)
    patches_collection = []
    #print('index list')
    #for item in index_lists:
    #    print(item)
    for index in index_lists:
        patch = tf.slice(image, index, patch_size)
        #patch = image[index[0]:(index[0] + patch_size[0]),index[1]: (index[1]+ patch_size[1]), index[2]:(index[2]+ patch_size[2])]
        patches_collection.append(patch)

    patches_collection = tf.stack(patches_collection)
    assert patches_collection.get_shape().dims == [num_patches, patch_size[0], patch_size[1], patch_size[2]]
    return patches_collection

########################################################################################################################
# @ author : Shanqi Yang
# input : the information about the image, and how it is supposed to be cropped
# output : lists of indexs list
# calls compute_patch_indices to calculate the index lists where the image will be cropped
########################################################################################################################

def compute_patch_indices(image_shape, patch_size, overlap, start = [0, 0, 0], order = True):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
        #print(overlap)

    stop = [(i-j) for i, j  in zip(image_shape, patch_size)]
    step = patch_size - overlap
    index_list = get_set_of_patch_indices(start, stop, step, order)
    return get_random_indexs (image_shape, patch_size, index_list)

# order is for fetching those near the bounds
# if fetch in True mode, then those near the stop won't be fetched
# if fetch in False mode , then those near the start won't be fetched

def get_set_of_patch_indices(start, stop, step, order = False):
    if order:
        return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                          start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
    else:
        return np.asarray(np.mgrid[stop[0]:start[0]:-step[0], stop[1]:start[1]:-step[1],
                           stop[2]:start[2]:-step[2]].reshape(3, -1).T, dtype=np.int)

def get_random_indexs (image_shape, patch_size, index_list):

    index0bound = image_shape[0] - patch_size[0]
    index1bound = image_shape[1] - patch_size[1]
    index2bound = image_shape[2] - patch_size[2]

    for index in index_list:
        newIndex0 = index[0] + random.randint(-10, 10)
        newIndex1 = index[1] + random.randint(-10, 10)
        newIndex2 = index[2] + random.randint(-10, 10)

        index[0] = newIndex0 if (newIndex0 <= index0bound and newIndex0 >= 0) else index[0]
        index[1] = newIndex1 if (newIndex1 <= index1bound and newIndex1 >= 0) else index[1]
        index[2] = newIndex2 if (newIndex2 <= index2bound and newIndex2 >= 0) else index[2]

    return index_list


def f3DPatching(img_origial, patch_size, overlap_rate):    
    # so2liu@gmail.com, a small subfunction 'pad_size_calculate' in this file is used.  
    # input: 3d np.array with zyx constellation, list with 3 elements in xyz constellation, a float
    # output: 4d np.array with (n, x, y, z) constellation
    
    # use z y x constellation
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
        patch_frame_shape = tuple(patch_frame_shape.astype(int))  # big patch struction
        result_shape = patch_frame_shape+tuple(patch_shape)
        result_strides = (sx*step_z*x*y, sx*x*step_y, sx*step_x, sx*x*y, sx*x, sx)    
        return np.lib.stride_tricks.as_strided(img, result_shape, result_strides)

  
    patched_img = fast_3D_strides(padded_img, patch_shape=patch_size, 
                                  stepsize_tuple=(patch_size[0], 
                                                  not_overlap_pixel_yx[0], 
                                                  not_overlap_pixel_yx[1]))  # axis2 is full not-overlap
    patched_img = np.reshape(patched_img, (-1, )+tuple(patch_size))

    return np.swapaxes(patched_img, 1, -1)  # return (n, x, y, z) constellation


def f3DUnpatching(patched_img, origin_shape, patch_size, overlap_rate):
    # so2liu@gmail.com
    # input: 4d imag with (n, x, y, z) as constellation, 1d oritin_shape (3 elements),  1d patch_size (3 elements), a float overlap_rate
    # output: non-padded, non-seperated, non-overlapped 3d img with size of patch_size in zyx constellation
    
    assert len(patched_img.shape) == 4 and isinstance(patch_size, list)
    
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

def pad_size_calculate(img_shape, not_overlap_shape, overlap_shape):
    # so2liu@gmail.com, used by my f3DPatching and f3DUnpatching subfunctions

    # a pad-calculation funciton, in z,y, x constellation
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