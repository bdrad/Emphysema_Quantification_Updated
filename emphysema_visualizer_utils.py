#!/usr/bin/env python
# coding: utf-8

import pydicom
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import cv2
from skimage import transform
import scipy
from ipywidgets import interact, IntSlider, HBox, VBox, Combobox, Checkbox, Dropdown, TwoByTwoLayout, Layout, interactive_output
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import display
import scipy.ndimage as ndi
from image_preprocessing_utils import window_image


def get_proper_ct_dict(lung_ct, axial_mask, emphysema_mask_dict):

    '''
    PARAMETERS:
    lung_ct: a 3d CT image windowed properly to the lung (axial as slice 0)
    axial_mask: a 3d lung segmentation mask (axial as slice 0)
    emphysema_mask_dict: a dictionary with the following key-value pair: {k:(i,j), v:(mask that highlights regions of emphysema)}
    '''
    ct_emphy= {}
    ct_emphy['LUNG'] = lung_ct
    ct_emphy['DEFAULT'] = lung_ct
    
    #Since -750 and 600 gives a min of -1050 and max of -450, we can safely translate the LUNG windowing with intercept 1 and slope 0
    ct_emphy['EMPHYSEMA DETECTION'] = window_image(lung_ct, -750, 600, 0, 1)
    ct_emphy['MASK'] = axial_mask
    ct_emphy['EMPHY MASK'] = emphysema_mask_dict

    return ct_emphy


# heavily modified based on show_plane and slice_explorer in NLST_utils
# 
# https://stackoverflow.com/questions/24124458/how-to-change-numpy-array-into-grayscale-opencv-image
# https://stackoverflow.com/questions/46020894/superimpose-heatmap-on-a-base-image-opencv-python
# https://stackoverflow.com/questions/63995578/change-colour-of-colorbar-in-python-matplotlib 
def show_plane(ax, plane, rescale=True, cmap="gray", mask_area = 0, emphy_mask=None, 
               img_min=None, img_max=None, title=None):
    """Shows a specific plane within 3D data.
       if rescale is True, stretch `plane` to be square looking
       if emphy_mask is not None, img_min and img_max must be provided 
       (these are the min HU value and max HU values of the patched ct slices)
    """
    
    if emphy_mask is None:
        if rescale:
            ax.imshow(plane, cmap=cmap, aspect = plane.shape[1]/plane.shape[0])
            # this converts plane to square looking
        else:
            ax.imshow(plane, cmap=cmap)
    else:

        plane = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
        if mask_area > 0: # this means that the img has already been masked 
            percentile = np.sum(emphy_mask > 0) / mask_area 
        if np.sum(emphy_mask) == 0: # all pixel value above threshold
            ax.imshow(plane, cmap=cmap)
        else:
            img_after_thres = cv2.cvtColor(emphy_mask, cv2.COLOR_GRAY2BGR)
            blur = cv2.GaussianBlur(img_after_thres,(3,3),3)
            heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_TURBO) # jet is not a good colormap

            heatmap_img[blur==0] = plane[blur==0] 
            # change regions where Gaussian Blur is black (no emphysema) to the `plane` pixel value so that
            # the final background image represented by `plane` is grayscale
            plane[heatmap_img > 0] = heatmap_img[heatmap_img > 0]
            # change_regions where `heatmap_img` is not black (emphysema) to `heatmap_img` pixel value so that
            # the final heatmap region brightness is preserved after blending
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, plane, 0.5, 0)
            heatmap = ax.imshow(super_imposed_img, vmin=img_min, vmax=-950, cmap='turbo')
            c = plt.colorbar(heatmap, ax=ax)
            # for displaying colorbar at proper scale
        if rescale:
            ax.axes.set_aspect(aspect = plane.shape[1]/plane.shape[0])
    '''
    ax.set_xticks([])
    ax.set_yticks([])
    '''

    if title:
        ax.set_title(title)
    if mask_area > 0 and emphy_mask is not None:
        print('Fraction Emphysema:', percentile)
    return None

def slice_explorer(datas, axis, cmap='gray'):
    """Allows to explore 2D slices in 3D data.
    Parameters
    ----------
    datas : dict: (key, value)
        key refers to a specific windowing kernel (window width(W), window level(L))
        four kernels are available:
        1. DEFAULT: based on original DICOM file
        2. LUNG: W:1500, L:-600
        3. MEDIASTINUM: W:350, L:50
        4. SOFT TISSUE: W:400, L:50 # easiest to evaulate organ in interest 
        value refers to the 3d ct image corresponding kernel
        
    axis: integer (0,1,or 2)
        The axis of 3d array perpendicular to sliced plane
    cmap : str (optional)
        A string referring to one of matplotlib's colormaps.
    """

    
    def display_slice(plane, WindowingOption, NumSlicesOption, ThresholdOption, PatchOption, MaskOption):
        
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111)
        #axis_3d = fig.add_subplot(122, projection='3d')
        
        
        if WindowingOption not in datas: # display DEFAULT windowing 
            data = datas['DEFAULT']
        else:
            data = datas[WindowingOption]
        
        masks = datas['MASK']
        
        if ThresholdOption:
            emphy_masks = datas['EMPHY MASK'][(ThresholdOption, PatchOption)]
            
        start = max(0, plane - NumSlicesOption)
        if axis == 0:
            img_slice = np.max(data[start:plane+1,:,:], axis=0) # axial MIP (0 is equivalent to no MIP)
            if ThresholdOption:
                emphy_mask = emphy_masks[plane,:,:]
            mask = masks[plane,:,:]

        elif axis == 1:
            img_slice = np.max(data[:,start:plane+1,:], axis=1)
            if ThresholdOption:
                emphy_mask = emphy_masks[:,plane,:]
            mask = masks[:,plane,:]
        elif axis == 2:
            img_slice = np.max(data[:,:,start:plane+1], axis=2)
            if ThresholdOption:
                emphy_mask = emphy_masks[:,:,plane]
            mask = masks[:,:,plane]
        
        if MaskOption:
            mask_area = np.sum(mask > 0)
            img_slice[mask == 0] = 0
        else:
            mask_area = 0
        
        #print(mask_area)
        if ThresholdOption:
            
            img_min, img_max = np.min(img_slice), np.max(img_slice)
            if img_min == img_max:
                show_plane(ax, img_slice, title='Plane {}'.format(int(plane)), cmap=cmap)
            else:
                emphy_mask = (emphy_mask*255.0)
                emphy_mask = emphy_mask.astype('uint8')
                rescaled_img_slice = (img_slice - img_min)/(img_max-img_min)*255.0
                rescaled_img_slice = rescaled_img_slice.astype('uint8') # for opencv image reading
                show_plane(ax, rescaled_img_slice, title='Plane {}'.format(int(plane)), cmap=cmap, 
                           emphy_mask = emphy_mask,
                           img_min=img_min,
                           img_max=img_max,
                          mask_area=mask_area) # rescale thres to (0, 255)
        else:
            show_plane(ax, img_slice, title='Plane {}'.format(int(plane)), cmap=cmap)

        #slice_in_3d(ax=axis_3d, axis=axis, shape=data.shape, plane=plane)
        
        
        
    plane = IntSlider(min=0, max=datas['DEFAULT'].shape[axis]-1, step=1, 
                                      value=datas['DEFAULT'].shape[axis]//2,
                      description = 'plane',
                      style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    WindowingOption = Combobox(options=['DEFAULT','LUNG','MEDIASTINUM','SOFT TISSUE',
                                       'EMPHYSEMA DETECTION'],  
                               # emphysema detection window level:-750, window width: 600
                               placeholder='Choose an option', 
                               description = 'windowing',
                               ensure_option=True, disabled=False,
                               style = {'description_width': 'initial'},
                               layout = Layout(width = '50%')
                               )
    NumSlicesOption = IntSlider(min=0, max=30, step=1, 
                                      value=0,
                      description = 'MIP slices',
                      style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    ThresholdOption = Dropdown(options = [False]+list(range(-850,-960,-10)),
                               value=False, 
                               description='Apply Threshold', 
                              style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    PatchOption = Dropdown(options = [3,5],
                               value=5, 
                               description='Patch Size', 
                              style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    
    MaskOption = Checkbox(value=False,
                        description='Apply Lung Segmentation Mask',
                        disabled=False,
                        indent=False,
                        style = {'description_width': 'initial'},
                        layout = Layout(width = '50%'))
    
    ui = VBox([TwoByTwoLayout(
               top_left=plane, 
               top_right=WindowingOption,
               bottom_left=NumSlicesOption,
               bottom_right=ThresholdOption),
              HBox([PatchOption, MaskOption])])

    out = interactive_output(display_slice, {'plane':plane,
                                             'WindowingOption': WindowingOption,
                                            'NumSlicesOption':NumSlicesOption,
                                             'ThresholdOption':ThresholdOption,
                                             'PatchOption': PatchOption,
                                            'MaskOption':MaskOption})
    
    display(ui, out)




