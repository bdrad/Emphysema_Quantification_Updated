#!/usr/bin/env python
# coding: utf-8

# ### Please cite this notebook as well as any references listed in this notebook if any function of this notebook is used.


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





# The following are util functions that can help in the preprocessing and/or visualization pipeline on raw CT images in DICOM format.


# Reference: https://www.kaggle.com/code/redwankarimsony/ct-scans-dicom-files-windowing-explained/notebook
# Function to take care of the translation and windowing. 
def window_image(original_img, window_center,window_width, intercept, slope, rescale=False):
    # img is a np array (2d or 3d)
    img = original_img.copy()
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    # data is a DICOM image
    try:
        dicom_fields = [data[('0028','1050')].value, #window center
                        data[('0028','1051')].value, #window width
                        data[('0028','1052')].value, #intercept
                        data[('0028','1053')].value] #slope
        return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    except:
        dicom_fields = [data[('0028','1052')].value, #intercept
                        data[('0028','1053')].value] #slope
        return [0, float('inf')] + [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def convert_images(dcmfiles, downsample_range = None, windowing=True, rescale=False):
    '''
    PARAMS:
    
        dcmfiles: A list of DICOM files
        downsample_range: a tuple (x,y) that specifies the bounding box of image for downscale operation
        rescale: whether to scale image to (0,255)
    
    RETURN:
        
        a list of tuples that contain the windowed images, as well as their relative z positions
    '''
    
    converted_images = []
    
    for im in range(len(dcmfiles)):
        
        data = dcmfiles[im]
        try:
            image_orientation = data[('0020','0037')].value
        except:
            continue # in rare cases, some of files in the series have no such info, discard
        image = data.pixel_array
        if image.shape[0] != 512 or image.shape[1] != 512:
            continue # in rare cases, some of files in the series have a different shape, discard
        if windowing:
            window_center, window_width, intercept, slope = get_windowing(data)
            output = window_image(image, window_center, window_width, intercept, slope, rescale = rescale)
        else:
            output = image
        if downsample_range is not None:
            output = transform.downscale_local_mean(output, downsample_range)
                    
        converted_images.append([output, 
                                image_orientation,
                                data[('0020','0032')].value[-1]
                                ]) # windowed_img, image orientation, image z position
    
    if len(converted_images) == 0:
        raise ValueError('no valid images in the series')
    
    return converted_images

def get_spacings(dcmfile):
    
    '''
    PARAMS:
    
        dcmfile: A single DICOM file
    
    RETURN:
        
        (z spacing, x spacing, y spacing)
    '''
    # single dcmfiles
    return [float(dcmfile[('0018','0050')].value), 
            float(dcmfile[('0028','0030')].value[0]), 
            float(dcmfile[('0028','0030')].value[1])]

# Reference: https://github.com/bdrad/high_precision_nodule_detector
def create_mip(img, num_slices_for_mip = 1, direction = 'axial'):
    '''
    Maximum Intensity Projection on `img` with `num_slices_for_mip` slices along `direction` direction
    '''
    
    mip_img = np.zeros(img.shape)
    if direction == 'axial':
        for i in range(img.shape[0]):
            start = max(0, i - num_slices_for_mip)
            mip_img[i,:,:] = np.max(img[start:i+1,:,:], axis=0)
    elif direction == 'coronal':
        for i in range(img.shape[1]):
            start = max(0, i - num_slices_for_mip)
            mip_img[:,i,:] = np.max(img[:,start:i+1,:], axis=1)
    elif direction == 'sagittal':
        for i in range(img.shape[2]):
            start = max(0, i - num_slices_for_mip)
            end = start + num_slices_for_mip
            mip_img[:,:,i] = np.max(img[:,:,start:i+1], axis=2)
    return mip_img

def create_thresholding(img, patch_size = 1, thres=-950): 
    # create threshold according to cube_sized patches (e.g. the center pixel is thresholded according to the average of the patch )
    # this is an out-of-date version of 2D kernel convolution
    # see create_threshold_3d_faster and 2d_faster in data_processing_utils.py
    '''
    thres value should be in the same unit as the unit of img pixel
    (i.e. either both in HU unit or both rescaled to (0,255))
    
    patch_size should ideally be odd
    
    RETURNS: (img rescaled to (0,255), patched_avgs rescaled to (0,255), img_min, img_max)
    '''
    print('Patch Size:', patch_size, 'Threshold Value:', thres)
    img_min, img_max = np.min(img), np.max(img)
    dist = patch_size // 2 # maximum perpendicular distance of patch from center to perimeter
    
    patch_avgs = np.zeros(img.shape)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            xmin = max(0, i-dist)
            xmax = min(img.shape[0], i+dist+1)
            ymin = max(0, j-dist)
            ymax = min(img.shape[1], j+dist+1)
            
            patch_avgs[i,j] = np.mean(img[xmin:xmax, ymin:ymax])
    
     
    return ((img-img_min)/(img_max-img_min)*255.0, (patch_avgs-img_min)/(img_max-img_min)*255.0,
           img_min, img_max) # rescale to (0,255)

# Reference: https://github.com/bdrad/high_precision_nodule_detector
def resize_image(img, spacing):
    RESIZE_SPACING = np.array([1, 1, 1])
    spacing = np.array(spacing)
    resize_factor = spacing / RESIZE_SPACING
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / img.shape
    new_spacing = spacing / real_resize

    #resize image
    '''
    e.g.: To use an image with 3 times fewer voxels along each dimension than our original version. 
    To keep the image bounds constant, the image spacing needs to be 3 times larger than the original one, 
    i.e., 3 mm. Our final image will contain around $3^3$ times fewer voxels than the full-resolution one!
    '''
    lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
    return lung_img, new_spacing

    
def sort_and_orient_slice_by_position(slices):
    
    '''
    PARAM: list of lists: [windowed_img, image orientation, image z position]
    RETURN: list of np arrays sorted by their z positions
    '''
    orientation = np.array(slices[0][1])
    slices = sorted(slices, key = lambda s: float(s[2]),
                   reverse=True)
    slices = np.array([s[0] for s in slices])
    # z-axis, y-axis, x-axis = axis 0, 1, 2
    # https://gist.github.com/agirault/60a72bdaea4a2126ecd08912137fe641
    if np.all(orientation == [1,0,0,0,1,0]):
        slices = slices
    elif np.all(orientation == [-1,0,0,0,-1,0]):
        slices = np.flip(np.flip(slices,1),2) # rotate 180 degree about z axis
    elif np.all(orientation == [-1,0,0,0,1,0]):
        slices = np.flip(np.flip(slices,0),2) # rotate 180 degree about y axis
    elif np.all(orientation == [1,0,0,0,-1,0]):
        slices = np.flip(np.flip(slices,0),1) # rotate 180 degree about x axis
    elif np.all(orientation == [0,1,0,-1,0,0]):
        slices = np.rot90(slices,-1, (2,1)) # clockwise rotate 90 degree about z axis
    elif np.all(orientation == [0,-1,0,1,0,0]): 
        slices = np.rot90(slices,1, (2,1))  # counterclockwise rotate 90 degree about z axis   
    elif np.all(orientation == [0,1,0,1,0,0]):
        for i in range(len(slices)):
            slices[i] = np.rot90(np.flip(slices[i],0), 1, (1,0))
        # flip about y axis (do not flip z axis) then clockwise rotate 90 degree about z axis  
    elif np.all(orientation == [0,-1,0,-1,0,0]):
        for i in range(len(slices)):
            slices[i] = np.rot90(np.flip(slices[i],0), -1, (1,0)) # flip about y axis (do not flip z axis) then counterclockwise rotate 90 degree about z axis 
    
    elif np.all(np.abs(np.array(orientation)) == [1,0,0,0,0,1]):
        raise ValueError('Localizer Image')
    else:
        slices = slices
       
    return slices

def get_3d_image(slices, downsample_range = None, windowing = True):
    
    '''
    PARAMS:
        slices: a list of DICOM files
        downsample_range
        windowing: whether to window according to the values stored in DICOM entry
    
    RETURN:
        3D numpy array where each slice along axis 0 is the axial slices
    '''
    slices = convert_images(slices, downsample_range, windowing=windowing)
    slices = sort_and_orient_slice_by_position(slices)
   
    img_3d = np.array(slices)
    return img_3d


# The following are visualization functions that can help visualize a 3d ct images in either static or dynamic fashion.

# heavily modified based on https://github.com/alexdesiqueira/ccb_skimage3d_tutorial/blob/main/supplementary_code.py

# https://stackoverflow.com/questions/24124458/how-to-change-numpy-array-into-grayscale-opencv-image
# https://stackoverflow.com/questions/46020894/superimpose-heatmap-on-a-base-image-opencv-python
# https://stackoverflow.com/questions/63995578/change-colour-of-colorbar-in-python-matplotlib
def show_plane(ax, plane, rescale=True, cmap="gray", thresholded_params=None, title=None):
    """Shows a specific plane within 3D data.
       if rescale is True, stretch `plane` to be square looking
       if provided a thresholded_params as a length-4 tuple, a heatmap will be plotted
    """
    
    if thresholded_params is None:
        if rescale:
            ax.imshow(plane, cmap=cmap, aspect = plane.shape[1]/plane.shape[0])
            # this converts plane to square looking
        else:
            ax.imshow(plane, cmap=cmap)
    else:
        
        img_before_thres, thres, patch_size, img_min, img_max = thresholded_params
        patch_size = 2*(patch_size // 2) + 1 # convert to odd for Gaussian kernel blurring
        thres_rescaled = (thres-img_min)/(img_max-img_min)*255.0
        img_before_thres = cv2.cvtColor(img_before_thres, cv2.COLOR_GRAY2BGR)
        plane = cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
        img_before_thres = cv2.cvtColor(img_before_thres, cv2.COLOR_BGR2GRAY) # grayscale image
        img_after_thres = cv2.threshold(img_before_thres,thres_rescaled,255,cv2.THRESH_BINARY)[1] 
        
        if np.all(img_after_thres > 0): # all pixel value above threshold
            ax.imshow(plane, cmap=cmap)
        else:
            img_after_thres = cv2.cvtColor(img_after_thres, cv2.COLOR_GRAY2BGR) # BGR image
            blur = cv2.GaussianBlur(img_after_thres,(patch_size,patch_size),np.sqrt(patch_size))
            heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_TURBO) # jet is not a good colormap

            heatmap_img[blur==255] = plane[blur==255] 
            # change regions where Gaussian Blur is white to the `plane` pixel value so that
            # the final background image represented by `plane` is grayscale
            plane[heatmap_img > 0] = heatmap_img[heatmap_img > 0]
            # change_regions where `heatmap_img` is not black to `heatmap_img` pixel value so that
            # the final heatmap region brightness is preserved after blending
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, plane, 0.5, 0)
            heatmap = ax.imshow(super_imposed_img, vmin=img_min, vmax=thres, cmap='turbo')
            c = plt.colorbar(heatmap, ax=ax)
            # for displaying colorbar at proper scale
        if rescale:
            ax.axes.set_aspect(aspect = plane.shape[1]/plane.shape[0])
         
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    return None




# modified based on https://github.com/alexdesiqueira/ccb_skimage3d_tutorial/blob/main/supplementary_code.py
# added codes so that all 3 planes of 3d array can be visualized

def slice_in_3d(ax, axis, shape, plane):
    """Draws a cube in a 3D plot.
    Parameters
    ----------
    ax : matplotlib.Axes
        A matplotlib axis to be drawn.
    axis: integer (0,1,or 2)
        The axis of 3d array perpendicular to sliced plane
        Axial correspind to 0
        Coronal correspond to 1
        Sagittal correspond to 2
    shape : tuple or array (1, 3)
        Shape of the input data.
    plane : int
        Number of the plane to be drawn.
    Notes
    -----
    Originally from:
    https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
    """
    Z = np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1]])

    Z = Z * shape

    r = [-1, 1]

    X, Y = np.meshgrid(r, r)

    # plotting vertices
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]],
             [Z[2], Z[3], Z[7], Z[6]]]

    # plotting sides
    ax.add_collection3d(
        Poly3DCollection(verts,
                         facecolors=(0, 1, 1, 0.25),
                         linewidths=1,
                         edgecolors='darkblue')
    )
    
    if axis == 0:
        verts = np.array([[[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [0, 1, 0]]]) # this is a axial plane square
        verts = verts * shape
        verts += [plane, 0, 0]
    elif axis == 1:
        verts = np.array([[[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 1],
                           [0, 0, 1]]]) # this is a coronal plane square
        verts = verts * shape
        verts += [0, plane, 0]
    elif axis == 2:
        verts = np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [1, 1, 0],
                           [1, 0, 0]]]) # this is a sagittal plane square
        verts = verts * shape
        verts += [0, 0, plane]

    ax.add_collection3d(
        Poly3DCollection(verts,
                         facecolors='magenta',
                         linewidths=1,
                         edgecolors='black')
    )

    ax.set_xlabel('axial slice')
    ax.set_ylabel('coronal slice')
    ax.set_zlabel('sagittal slice')

    # auto-scale plot axes
    scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)

    return None


# In[6]:


# heavily modified based on https://github.com/alexdesiqueira/ccb_skimage3d_tutorial/blob/main/supplementary_code.py

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

    
    def display_slice(plane, WindowingOption, NumSlicesOption, ThresholdOption, PatchOption):
        
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(121)
        axis_3d = fig.add_subplot(122, projection='3d')
        
        
        if WindowingOption not in datas: # display DEFAULT windowing 
            data = datas['DEFAULT']
        else:
            data = datas[WindowingOption]
        
        start = max(0, plane - NumSlicesOption)
        if axis == 0:
            img_slice = np.max(data[start:plane+1,:,:], axis=0) # axial MIP (0 is equivalent to no MIP)
        elif axis == 1:
            img_slice = np.max(data[:,start:plane+1,:], axis=1)
        elif axis == 2:
            img_slice = np.max(data[:,:,start:plane+1], axis=2)
       
        if ThresholdOption:
            rescaled_img_slice, patched_img_slice, img_min, img_max = create_thresholding(img_slice, PatchOption, ThresholdOption)
            rescaled_img_slice = rescaled_img_slice.astype('uint8') # for opencv image reading
            patched_img_slice = patched_img_slice.astype('uint8')
            show_plane(ax, rescaled_img_slice, title='Plane {}'.format(int(plane)), cmap=cmap, 
                       thresholded_params=(patched_img_slice, 
                                           ThresholdOption,
                                           PatchOption,
                                            img_min, img_max)) # rescale thres to (0, 255)
        else:
            show_plane(ax, img_slice, title='Plane {}'.format(int(plane)), cmap=cmap)

        slice_in_3d(ax=axis_3d, axis=axis, shape=data.shape, plane=plane)
        
        fig.tight_layout()
        plt.show()
        
    plane = IntSlider(min=0, max=datas['DEFAULT'].shape[axis]-1, step=1, 
                                      value=datas['DEFAULT'].shape[axis]//2,
                      description = 'plane',
                      style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    WindowingOption = Combobox(options=['DEFAULT','LUNG','MEDIASTINUM','SOFT TISSUE'], 
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
    
    ThresholdOption = Dropdown(options = [False,
                                          -850,
                                          -900,
                                          -950],
                               value=False, 
                               description='Apply Threshold', 
                              style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    PatchOption = IntSlider(min=1, max=9, step=1, 
                                      value=1,
                      description = 'Patch Size (Thresholding)',
                      style = {'description_width': 'initial'},
                     layout = Layout(width = '50%'))
    
    ui = VBox([TwoByTwoLayout(
               top_left=plane, 
               top_right=WindowingOption,
               bottom_left=NumSlicesOption,
               bottom_right=ThresholdOption),
              PatchOption])

    out = interactive_output(display_slice, {'plane':plane,
                                             'WindowingOption': WindowingOption,
                                            'NumSlicesOption':NumSlicesOption,
                                             'ThresholdOption':ThresholdOption,
                                            'PatchOption':PatchOption})
    
    display(ui, out)
    



def slice_comparer(data1, data2, axis, cmap='gray'): 
    """Allows to compare between 2D slices of two different 3D data.
    Parameters
    ----------
    data1, data2 : array (M, N, P)
        3D interest image.
    axis: integer (0,1,or 2)
        The axis of 3d array perpendicular to sliced plane
    cmap : str (optional)
        A string referring to one of matplotlib's colormaps.
    """
    assert data1.shape == data2.shape, 'The two 3d image must have the same shape.'
    @interact(plane=(0, data1.shape[axis]-1), continuous_update=False)
    def display_slice(plane=data1.shape[axis]//2):
        
        fig, axes = plt.subplots(2,1,figsize=(20, 14))
        axis_3d1 = fig.add_subplot(233, projection='3d')
        axis_3d2 = fig.add_subplot(236, projection='3d')
        
        if axis == 0:
            show_plane(axes[0], data1[plane,...], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
            show_plane(axes[1], data2[plane,...], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
        elif axis == 1:
            show_plane(axes[0], data1[:,plane,:], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
            show_plane(axes[1], data2[:,plane,:], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
        elif axis == 2:
            show_plane(axes[0], data1[...,plane], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
            show_plane(axes[1], data2[...,plane], rescale = True, title='Plane {}'.format(int(plane)), cmap=cmap)
        slice_in_3d(ax=axis_3d1, axis=axis, shape=data1.shape, plane=plane)
        slice_in_3d(ax=axis_3d2, axis=axis, shape=data1.shape, plane=plane)
        #fig.tight_layout()
        plt.show()

    return display_slice
