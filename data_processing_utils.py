#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd 
import math
import pydicom
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, remove_small_holes, erosion, closing, binary_closing, convex_hull_image
from skimage.measure import label,regionprops, perimeter, find_contours
from skimage.morphology import binary_dilation, binary_opening
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from skimage.transform import resize, warp, AffineTransform
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import glob
from skimage.io import imread
from image_preprocessing_utils import *

'''
IN GENERAL, NEED TO ADJUST PARAMETERS FOR IMAGE SIZE WHEN NEEDING TO DETECT TRACHEA, EXCLUDE THEM, OR FILL UP HOLES DUE TO FALSE POSITIVE REMOVAL OF TRACHEA_LIKE AREA, HENCE FILL_LARGE_HOLE_THRES AND FILL_SMALL_HOLE_THRES SHOULD NOT REQUIRE ADJUSTMENT SINCE THEY SERVE TO GENERALLY FILL UP LUNG HOLES

ADJUST AREA THRESHOLD BY (w/512)**2 because removing border does not make sense when given a lung mask as the input
'''

DIFF_TO_MIDDLE = 0.06
PERCENT_OVERLAP = 0.8
TRACHEA_REMOVAL_AREA = 200
MIN_LUNG_AREA = 1000
SPINDLYNESS_INDEX_THRES = 4
MIN_BORDER_WIDTH_RATIO = 0.8
MIN_BORDER_CENTROID_RATIO = 0.7
FILL_LARGE_HOLE_THRES = 50 # small enough area and not involved in trachea detection, no need to adjust for image dimension
FILL_SMALL_HOLE_THRES = 10 # small enough area and not involved in trachea detection, no need to adjust for image dimension
ANGLE_THRES = 5/180

def get_width_at_each_height(region_mask):
    '''
    returns the max width at each x value of 2d image represented by a 2d array
    '''
    xs, ys = np.where(region_mask > 0)
    unique_heights = np.unique(xs)
    width_at_each_height = []
    for height in unique_heights:
        indices = np.where(xs==height)[0]
        width_at_each_height.append(max(ys[indices])-min(ys[indices])+1)
    return width_at_each_height

def get_height_at_each_width(region_mask):
    '''
    returns the max height at each y value of 2d image represented by a 2d array
    '''
    xs, ys = np.where(region_mask > 0)
    unique_widths = np.unique(ys)
    height_at_each_width = []
    for width in unique_widths:
        indices = np.where(ys==width)[0]
        height_at_each_width.append(max(xs[indices])-min(xs[indices])+1)
    return height_at_each_width

def spindlyness_index(region_mask):
    '''
    can be thought of a relative measure of how thin/spindly a region is
    '''
    width_at_each_height = get_width_at_each_height(region_mask)
    height_at_each_width = get_height_at_each_width(region_mask)
    return max(len(width_at_each_height)/(max(width_at_each_height)),
                  len(height_at_each_width)/(max(height_at_each_width)))
    
def remove_borders(raw_im, very_superior_end=False):
    '''
    Input (raw_im): a 2D axial slice windowed according to the lung window level and width
    Input (very_superior_end): TRUE iff the first 1/10 slices (from superior to inferior)
    Output: a binarized 2D image with border removed
    '''
    h, w = raw_im.shape
    
    def normal_fill_border_mask(img): 
        # fill all circular border so that any lung is not cleared
        new_img = img.copy()
        h,w = img.shape
        assert h == w
        r = h/2
        x = np.linspace(0, h, h)
        y = np.linspace(0, w, w)
        xv, yv = np.meshgrid(x, y)
        xv,yv = xv-h/2, yv-h/2
        border_mask = xv**2+yv**2 > (r+1)**2
        new_img[border_mask] = 0
        return new_img
    
    def remove_borders_first_pass(binary):
        label_image = label(binary, background=0, connectivity=1) # separate border fragments more completely
        h,w = binary.shape
        border_coords = set([(0,0), (h-1,0), (0,w-1), (h-1,w-1), (0,w//2), (h-1, w//2)])
        for r in regionprops(label_image):
            remove_region = False
            if r.centroid[0]/h > (MIN_BORDER_CENTROID_RATIO+0.1): # this is specifically for axial slices
                remove_region = True

            for x,y in border_coords:
                if np.any(r.coords[:,0] == x) and np.any(r.coords[:,1]==y):
                    remove_region = True
                    break

            if remove_region:
                for coordinates in r.coords:
                    binary[coordinates[0], coordinates[1]] = 0
        return binary
    
    def remove_borders_second_pass(binary, very_superior_end=False):
        '''
        remove small objects (such as airways)
        '''
        h, w = binary.shape
        label_image = label(binary, background=0, connectivity=1)
        if very_superior_end:
            binary = remove_small_objects(binary, min_size=TRACHEA_REMOVAL_AREA*(w/512)**2) # lungs too small, so use trachea area as lung area threshold
        else:
            binary = remove_small_objects(binary, min_size=MIN_LUNG_AREA*(w/512)**2) # remove airways to its fullest
        binary = remove_small_holes(binary, area_threshold=FILL_LARGE_HOLE_THRES) # always be careful of introducing new artifacts after closing/removing holes
        label_image = label(binary, background=0)
        for r in regionprops(label_image, extra_properties=[spindlyness_index]):
            xs, ys = r.coords[:,0], r.coords[:,1]
            if np.max(xs)-np.min(xs) < w/20 and r.spindlyness_index > SPINDLYNESS_INDEX_THRES:
                for c in r.coords:
                    binary[c[0],c[1]]=0
            elif np.max(ys)-np.min(ys) > w*MIN_BORDER_WIDTH_RATIO  and r.centroid[0] > MIN_BORDER_CENTROID_RATIO :
                for c in r.coords:
                    binary[c[0],c[1]]=0
        return binary
    
    binary = raw_im < -400
    binary = remove_borders_first_pass(binary)
    binary = remove_borders_second_pass(binary, very_superior_end)
    binary = normal_fill_border_mask(binary)
    return binary


def get_segmented_lungs_trial_and_error(raw_im, return_mask=False, superior_end=False, inferior_end = False, plot=False):
    
    '''
    this is a unsupervised segmentation algorithm on axial slice, might not be able to accurately segment lung for every given
    2d lung image but should do a decent job on most axial slice with full view of both lungs 

    INPUT (raw_im): output of remove_borders
    INPUT (superior_end): TRUE iff the first 1/4 slices (from superior to inferior)
    INPUT (inferior_end): TRUE iff the last 1/4 slices (from superior to inferior)
    OUTPUT: a segmented lung mask (left and right lobes not separated)
    '''
    
    
    def is_symmetrical(region1, region2):
        c1 = region1.centroid
        c2 = region2.centroid
        return abs(c1[0]-c2[0])/(h-top_border_width-bottom_border_width) < DIFF_TO_MIDDLE and \
                abs((c1[1]+c2[1])/(w+2*padding) - 1) < DIFF_TO_MIDDLE and \
                region1.area/region2.area < 5
    
    def lung_region_cond(c):
        
        return ((c[0]-top_border_width-padding)/new_h < 1/3 or \
                            ((c[1]-padding)/new_w > (0.5-DIFF_TO_MIDDLE) and (c[1]-padding)/new_w < (0.5+DIFF_TO_MIDDLE)))
                
    def label_and_select_regions(img):
        '''
        img_area, h, w, top_border_width, bottom_border_width, left_border_width, right_border_width, padding,
        new_h, new_w defined in main function body
        
        used only when two lung regions need to be selected from the img
        '''
        label_image = label(img)
        valid_regions = [r for r in regionprops(label_image, extra_properties=[spindlyness_index]) if r.area > MIN_LUNG_AREA*(w/512)**2] #this is for removing small objects such as trachea
        valid_regions = sorted(valid_regions, key=lambda r: r.area)
        if len(valid_regions) == 0:
            return label_image, []
        original_valid_regions = valid_regions
                
        i = len(valid_regions) - 1
        while i > -1: # find the largest valid region
            c = valid_regions[i].centroid
            region1_cond = lung_region_cond(c) and valid_regions[i].area < img_area/20
            '''
            # if the region is a lung, it is either in the middle regions, where its area should not be too small,
            # or it is in the periphery, where its centroid should not be in the upper 1/3 or in the middle
            # in terms of horizontal(unlike the trachea)
            
            h-top_border_width-bottom_border_with is the height of the body portion
            w is the original image width, but centroid position includes padding so need to remove it
            '''
            
            if region1_cond:
                i -= 1 # continue find the next largest region
            else:
                valid_regions = valid_regions[:i+1] # remove the previous larger regions
                break
        if i == -1: 
            '''
            if no valid region exist, then it could be that that the two small lung lobes at
            the early periphery axial slice that somehow satisfies region1_cond 
            so that they are skipped.
            
            Attempt to refind 2 roughly symmetrical objects with respect to the middle vertical line across 
            the center of the image by the following:
            
            ''' 
            if len(original_valid_regions) >= 2:
                region1 = original_valid_regions[-1]
                region2 = original_valid_regions[-2]
                if is_symmetrical(region1, region2):
                    return label_image, [r.area for r in original_valid_regions[-2:]]
            return label_image, []
        largest_region_i = i
        i -= 1 # transit from largest region to second largest
        while i > -1:
            if ((valid_regions[i].centroid[1]-padding)/new_w > 0.5 and (valid_regions[-1].centroid[1]-padding)/new_w > 0.5) or \
            ((valid_regions[i].centroid[1]-padding)/new_w < 0.5 and (valid_regions[-1].centroid[1]-padding)/new_w < 0.5): # both left or both right, not accepted
                # both left or both right, not accepted
                i -= 1
            else:
                region2_area = valid_regions[i].area
                region1_area = valid_regions[-1].area
                c = valid_regions[i].centroid
                region2_cond = lung_region_cond(c) and valid_regions[i].area < img_area/20
                # same reasoning as region1_cond
                if region2_cond or region2_area < region1_area/50: 
                    # the second condition add as a safety measure for removing trachea 
                    i -= 1
                else:
                    break
        fragment_cond = lambda r: r.area<img_area/50 and r.spindlyness_index<SPINDLYNESS_INDEX_THRES # further selecting for gramenets such as trachea
        if i == -1:
            valid_regions = [r for r in valid_regions if not fragment_cond(r)]            
        else:
            valid_regions = [r for r in valid_regions[:i]+valid_regions[i+1:-1] if not fragment_cond(r)]+[valid_regions[i]]+valid_regions[-1:]
            # rearrange so that valid_regions[i+1] is arranged at 2nd place, and remove potential fragments
            region1 = original_valid_regions[-1]
            region2 = original_valid_regions[-2]
            c1 = region1.centroid
            c2 = region2.centroid
        
        valid_areas = [r.area for r in valid_regions]
        return label_image, valid_areas
                        
    
    def erode_and_label(img,r):
        selem = disk(r)
        disconnected = erosion(img, selem)
        label_image, valid_areas = label_and_select_regions(disconnected)
        return label_image, disconnected, valid_areas
    
    def close_and_label(binary_img,r):
        selem = disk(r)
        filled = binary_closing(binary_img, selem)
        label_image, valid_areas = label_and_select_regions(filled)
        return filled, valid_areas

            
    im=raw_im.copy()
    h, w = im.shape
    original_w = w

    '''
    This function segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
        
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(raw_im, cmap = 'gray') 
        plots[0].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])
        
    '''
    Step 1: Convert into a binary image. 
    (already done by previous steps)
    '''
    binary = im
    if np.all(binary == 1):
        return np.zeros(im.shape)

    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(binary, cmap='gray') 
        plots[1].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])  

    '''
    Step 2: Remove border (done by previous steps) and update border width
    '''
    xs, ys = np.nonzero(1-binary.astype('int')) # the outer regions become 1 and body becomes 0, need to invert to use np.nonzero properly
    top_border_width = min(xs) # account for the part above the body
    bottom_border_width = h-max(xs)
    left_border_width = min(ys)
    right_border_width = w-max(ys)
    img_area = (max(xs)-min(xs))*(max(ys)-min(ys))
    new_h, new_w = h-top_border_width-bottom_border_width, w 
              
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(binary, cmap='gray')
        plots[2].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])
          
    cleared = np.pad(binary, ((20,20),(20,20)),
                       'constant',constant_values=0)
    padding = 20 # update padding

    '''
    Step 3: Disconnect trachea and Label the image.
    '''
    label_image, areas = label_and_select_regions(cleared)
    if len(areas) == 0:
        return np.zeros(raw_im.shape)
    else:
        label_image = label(cleared)
        valid_regions = [r for r in regionprops(label_image) if r.area > 10] #this is for removing extremely small objects
        valid_regions = sorted(valid_regions, key=lambda r: r.area)
        if len(valid_regions) >=3 and valid_regions[-2].area > valid_regions[-1].area/2 and valid_regions[-1].area > img_area/20 \
           and valid_regions[-3].area > MIN_LUNG_AREA*(w/512)**2: # last condition is to exclude trachea
            c = valid_regions[-3].centroid
            region3_cond = lung_region_cond(c) and valid_regions[-3].area < img_area/20 # this region3_cond is True when region3 looks like a trachea
            old_areas = [r.area for r in valid_regions]
            if not region3_cond:
                new_binary = binary_closing(cleared, disk(2)) # glue together pieces of lungs eroded by thresholding
                new_cleared = remove_small_holes(new_binary, area_threshold=FILL_SMALL_HOLE_THRES)
                label_image, areas = label_and_select_regions(new_cleared)
                if areas[-1] > old_areas[-1] + old_areas[-3]*0.9 or areas[-2] > old_areas[-2] + old_areas[-3]*0.9:
                    cleared = new_cleared
                else:
                    label_image, areas = label_and_select_regions(cleared)
    areas.sort()
    
    r = 1
    disconnected = cleared
    old_areas = areas
    while len(areas) == 1 and r < 5 and sum(areas[-2:]) >= old_areas[-1]*0.8: 
        '''
        most likely the lung is one intact piece of left and right lobe,
        gradually increase r of sgructuring element to erode lung regions
        separate into multiple connected regions, the second condition prevents overerosion
        '''
        label_image, disconnected, areas = erode_and_label(cleared,r)
        r += 1
    if len(areas)==1 or sum(areas[-2:]) < old_areas[-1]*0.8: 
        # if still one connected set or overerosion, then revert to the original image
        label_image, areas = label_and_select_regions(cleared)
        areas.sort()
    elif len(areas)>1:
        cleared = disconnected
        if len(areas) < 3 or areas[-2] <= areas[-1]/50 or areas[-3] <= areas[-2]/50: # try to determine if trachea is still present or is connected to lung
            r = 3
            old_areas = areas
            label_image, disconnected, areas = erode_and_label(cleared,r)
            if len(areas) <= 2 or (sum(areas[:-2]) <= areas[-1]/50 or sum(areas[-2:]) <= sum(old_areas[-2:])*2/3) or (len(areas)>=3 and areas[-3] <= areas[-2]/50):
                '''
                3 conditions connected by or
                first condition, if len(areas) <= 2, it might be no separation of trachea has occured
                second condition, composed of 2 sub-conditions (a) and (b)
                (a) ensures that the third largest area is not trachea because its area plus the second largest area is still too small
                (b) ensures that if the first condition is not satisfied, it is not due to artifact casued by over erosion of lung 
                third condition, maybe the thrid region is trachea because it is too small

                need further erosion to check
                '''
                r = 5 # maybe r=3 is not enough to separate trachea
                # the reason why do not start eith r = 5 is maybe it will overerode the trachea
                label_image, disconnected, areas = erode_and_label(cleared,r)
                if len(areas) == 0 or (sum(areas[:-2]) < areas[-1]/50 or sum(areas[-2:]) <= sum(old_areas[-2:])*2/3):
                    # if the effect is the same as r=3, then revert to non-eroded images
                    revert = True
                else:
                    revert = False
            else:
                revert = False
            
            if revert:
                # revert to uneroded
                label_image, areas = label_and_select_regions(cleared)

            else: 
                '''
                # keep reduce erosion to minimum radius such that trachea is separate from the lungs
                # the first condition ensures two lung area not very different
                # the second condition ensures trachea is still separated
                '''
                while (len(areas) >= 2 and areas[-2] > areas[-1]/2) and (len(areas) >= 3 and areas[-3] > areas[-2]/50) and r > 1: # areas[-3] is trachea
                    r-=1
                    label_image, disconnected, areas = erode_and_label(cleared,r)
                label_image, disconnected, areas = erode_and_label(cleared,r+1)
    
  
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(label_image, cmap='gray') 
        plots[3].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])
    
    '''
    Step 4: Keep the labels with 2 largest areas (so can serve to remove trachea in some cases). 
    sometimes 3 or 4 areas are kept because one or both lung lobe is composed of 2 connected regions instead of 1
    '''

    def half_lung_second_region_cond(r):
        return r.spindlyness_index >= SPINDLYNESS_INDEX_THRES and regions[-2].area/r.area < 10 #last condition set lower bound for a valid region'a area
    
    if len(areas) == 0:
        return np.zeros(raw_im.shape) # no valid regions
    
    elif len(areas) >= 3:
        regions = [r for r in regionprops(label_image, extra_properties=[spindlyness_index]) if r.area > 10]
        regions.sort(key=lambda r:r.area)
        
        region3_cond = half_lung_second_region_cond(regions[-3])
        if len(areas) >= 4:
            region4_cond = half_lung_second_region_cond(regions[-4])
        else:
            region4_cond = False
        if region3_cond and region4_cond:
            num_lung_regions = 4
        elif region3_cond:
            num_lung_regions = 3
        else:
            num_lung_regions = 2
    else:
        num_lung_regions = 2
    regions = regionprops(label_image)
    for region in regions:
        if region.area not in areas[-min(num_lung_regions,len(areas)):]:
            for coordinates in region.coords:                
                label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
        plots[4].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])

    if superior_end:
        # skip step 5
        binary = binary[20:-20,20:-20]
        
    else:
        '''
        Step 5: Closure operation with a disk of radius 10. This operation is to keep nodules attached to the lung wall. 
        But do not want to close the space between left and right lung, so need to decrease radius if necessary
        '''
        # pad again before closure to avoid closing lung with border, 
        # technically not necessary because the img is already padded,
        # can remove this
        binary = np.pad(label_image > 0, ((20,20),(20,20)),
                        'constant',constant_values=0)
        
        padding = 40 # update padding
        areas = [r.area for r in regionprops(label(binary))]
        
        r=10
        new_binary, new_areas = close_and_label(binary, r)
        while (len(new_areas) <= 1 or new_areas[-1] > 1.15*areas[-1]) and r > 2: 
            # make sure the space does not get filled by evaluating the growth of error
            r-=1
            new_binary, new_areas = close_and_label(binary, r)
        binary=new_binary
        binary = binary[40:-40,40:-40] # remove padding (two times 20) at the end
        padding = 0 # update padding
        if plot == True:
            plots[5].axis('off')
            plots[5].imshow(binary, cmap=plt.cm.bone) 
            plots[5].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])
    
    '''
    Step 6: Fill in the small holes inside the binary mask of lungs
    '''
    binary = remove_small_holes(binary, area_threshold=FILL_LARGE_HOLE_THRES) # fill up fragmented lungs
    if superior_end:
        label_image = label(binary, connectivity=1) # remove airway
        regions = regionprops(label_image)
        print(len(regions))
        areas = [r.area for r in regions]
        areas.sort()
        if len(areas) <= 1:
            binary = label_image > 0
        else:
            list_is = []
            for i in range(1, np.max(label_image)+1):
                if np.sum(label_image) >= areas[-2]: # find top 2 areas
                    list_is.append(i)
            binary = np.logical_or(label_image == list_is[0], label_image ==list_is[1])

    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
        plots[6].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])

    
    '''
    Step 7: Final sanity check to remove trachea/fragments
    '''
    label_image, areas = label_and_select_regions(binary)
    regions = regionprops(label_image, extra_properties=[spindlyness_index])
    regions.sort(key=lambda r: r.area)
    
    # check if the second region is surrounded by the first region in at least 3 directions
    # i.e. check if region2 lies in the convex hull of region 1
    if len(regions) > 1:
        temp_binary = np.zeros(label_image.shape)
        for c in regions[-1].coords:
            temp_binary[c[0],c[1]] = 1
        temp_binary = convex_hull_image(temp_binary)
        if np.all([temp_binary[c[0],c[1]] > 0 for c in regions[-2].coords]):
            for c in regions[-2].coords:
                binary[c[0],c[1]]=0
            label_image, areas = label_and_select_regions(binary)
            regions = regionprops(label_image, extra_properties=[spindlyness_index])
            regions.sort(key=lambda r: r.area)
            
    
    candidate_regions = []
    for i in range(1,max(num_lung_regions,2)+1):
        if i > len(regions):
            break
        candidate_regions.append(regions[-i])
    
    '''
    separate regions according to whether they reside in left half or right half of the image
    '''
    label_1_regions = []
    label_2_regions = []
    
    label_image = np.zeros(label_image.shape)
    for region in candidate_regions:
        if region.centroid[1]/new_w < 0.5:
            label_1_regions.append(region)          
        else:
            label_2_regions.append(region)
    
    label_1_regions.sort(key=lambda r: max(r.coords[:,0]), reverse=True) # sort by lowest point (max x coord)
    label_2_regions.sort(key=lambda r: max(r.coords[:,0]), reverse=True)
    
    '''
    Now we remove improper regions by going at each side from down to up
    because the trachea/fragment regions usually lie in the upper part of the image
    '''

    def invalid_region_cond(prev_region, curr_region):
        lowest_prevregion_x = max(prev_region.coords[:,0])
        highest_prevregion_x = min(prev_region.coords[:,0])
        lowest_currregion_x = max(curr_region.coords[:,0])
        highest_currregion_x = min(curr_region.coords[:,0])
        #print(curr_region.spindlyness_index, highest_prevregion_x, lowest_currregion_x, highest_currregion_x)
        
        return (highest_prevregion_x-lowest_currregion_x) > (lowest_prevregion_x-highest_prevregion_x)*1/3 \
               and curr_region.spindlyness_index < SPINDLYNESS_INDEX_THRES
        '''  
        # the first condition means that the current region is too far away from the previous lowest region,
        # important when comparing the left lung and the right lung
        # the second conditon means that the current region is unlikely a crescent shaped lung
        '''
    if len(label_1_regions) > 1:
        prev_valid_i = 0 # we assume the lowest region is a valid lung lobe
        valid_label_1_regions = [label_1_regions[0]]
        for i in range(1, len(label_1_regions)):
            if not invalid_region_cond(label_1_regions[prev_valid_i], label_1_regions[i]):
                prev_valid_i = i # update the previous lowest region
                valid_label_1_regions.append(label_1_regions[i])
            else:
                for coordinates in label_1_regions[i].coords:
                    label_image[coordinates[0], coordinates[1]] = 0
        label_1_regions = valid_label_1_regions
    if len(label_2_regions) > 1:
        prev_valid_i = 0
        valid_label_2_regions = [label_2_regions[0]]
        for i in range(1, len(label_2_regions)):
            if not invalid_region_cond(label_2_regions[prev_valid_i], label_2_regions[i]):
                prev_valid_i = i
                valid_label_2_regions.append(label_2_regions[i])
            else:
                for coordinates in label_2_regions[i].coords:
                    label_image[coordinates[0], coordinates[1]] = 0
        label_2_regions = valid_label_2_regions
    
    if inferior_end: # all lung lobes is mainly in the lower half of the image
        new_label_1_regions = []
        for region in label_1_regions:
            if region.spindlyness_index < 2 and max(region.coords[:,0])/h < 0.5:
                continue
            else:
                new_label_1_regions.append(region)
        label_1_regions = new_label_1_regions
        new_label_2_regions = []
        for region in label_2_regions:
            if region.spindlyness_index < 2 and max(region.coords[:,0])/h < 0.5:
                continue
            else:
                new_label_2_regions.append(region)
        label_2_regions = new_label_2_regions
        
        
    if len(label_1_regions) == 1 and len(label_2_regions) == 1: 
        # remove non-lung fragment if that is the only region in left/right half plane
        if invalid_region_cond(label_2_regions[0], label_1_regions[0]):
            label_1_regions = []
        elif invalid_region_cond(label_1_regions[0], label_2_regions[0]):
            label_2_regions = []
    valid_regions = label_1_regions+label_2_regions
    
    binary = np.zeros(label_image.shape)
    for region in valid_regions:
        for coordinates in region.coords:
            binary[coordinates[0], coordinates[1]] = 1
           

    im[binary == 0] = 0
    
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap='gray') 
        plots[7].set_aspect(aspect = raw_im.shape[1]/raw_im.shape[0])
    if return_mask:
        return binary.astype('int')
    else:
        return im
    
    
def separate_lungs_2d(lung_mask_2d):
    '''
    separate the left and right lung lobes so that the left lung is labeled 1 and the right lung is labeled 2
    INPUT: 2d binary lung mask
    OUTPUT: 2d ternary lung mask
    '''
    h,w = lung_mask_2d.shape
    label_image = label(lung_mask_2d>0)
    old_label_image = label_image
    label_image = np.zeros(label_image.shape)
    for region in regionprops(label(old_label_image>0)):
        if region.centroid[1]/w < 0.5:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 2
        else:
            for coordinates in region.coords:
                label_image[coordinates[0], coordinates[1]] = 1
    return label_image

def postprocess_remove_airway(lung_ct_2d, lung_mask_2d, airway_thres=-800):
    '''
    remove airway regions from the lung mask based of thresholding
    '''
    new_lung_mask_2d = lung_mask_2d.copy()
    h,w = lung_ct_2d.shape
    
    label_lung_mask_2d = label(lung_mask_2d > 0)
    new_lung_mask_2d = lung_mask_2d.copy()
    temp_binary = np.logical_and(lung_ct_2d< airway_thres, lung_mask_2d>0)
    label_image = label(temp_binary, connectivity=1)
    label_masks = {j:label_image==j for j in range(1,np.max(label_image)+1)}
    candidate_js = [j for j in label_masks if np.sum(label_masks[j]) in range(int(TRACHEA_REMOVAL_AREA*(w/512)**2), int(MIN_LUNG_AREA*(w/512)**2))] # candidate trachea region
    for j in candidate_js:
        
        contours = find_contours(label_masks[j])
        old_area = np.sum(label_masks[j])
        # reference: https://stackoverflow.com/questions/39642680/create-mask-from-skimage-contour
        r_mask = np.zeros_like(lung_ct_2d, dtype='bool')
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        r_mask[np.round(contours[0][:, 0]).astype('int'), np.round(contours[0][:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        r_mask = ndi.binary_fill_holes(r_mask)
        new_area = np.sum(r_mask)
        if old_area/new_area >= PERCENT_OVERLAP: #this means the original region is one intact piece so should be airway
            best_i, overlap_area = 0, 0
            # find the original region in lung_mask_2d that this new region belongs to by the criterion of greatest overlap
            for i in range(1, np.max(label_lung_mask_2d)+1):
                curr_overlap_area = np.sum(np.logical_and(label_lung_mask_2d==i, r_mask))
                if curr_overlap_area > overlap_area:
                    best_i, overlap_area = i, curr_overlap_area
            original_lung = label_lung_mask_2d==best_i
            old_xs, old_ys = np.where(original_lung)
            new_xs, new_ys = np.where(r_mask == 1)
            x1, x2, x3, x4 = min(old_xs), max(old_xs), min(new_xs), max(new_xs)
            y1, y2, y3, y4 = min(old_ys), max(old_ys), min(new_ys), max(new_ys)
            if x1 > x3 and x2 < x4 and y1 > y3 and y2 > y4: # entirely contained within the lung, not airwa
                continue
            elif x4 < x1 + (x2-x1)/3 or x3 > x2 - (x2-x1)/3: # not in middle 2/3 of the lung
                continue
            else:
                unique_xs = np.unique(new_xs)
                unique_ys = np.unique(new_ys)
                num_ys_on_border = 0
                num_xs_on_border = 0
                
                if False:
                    continue
                else:
                    if min(old_ys) < 0.5: # left lung
                        for x in unique_xs:
                            if np.any(old_xs == x) and max(old_ys[old_xs == x]) <= max(new_ys[new_xs == x]): 
                                # find x coordinates on right border of left lung
                                num_xs_on_border += 1
                    elif max(old_ys) > 0.5: # right lung
                        for x in unique_xs: 
                            if np.any(old_xs == x) and min(old_ys[old_xs == x]) >= min(new_ys[new_xs == x]):
                                # find x coordinates on left border of right lung
                                num_xs_on_border += 1
                    if num_xs_on_border >= min(5, len(unique_xs)):
                        continue
                    
            if overlap_area/new_area > PERCENT_OVERLAP and overlap_area/np.sum(label_lung_mask_2d==best_i) > PERCENT_OVERLAP:
                new_lung_mask_2d[label_lung_mask_2d==best_i] = 0
            else:    
                new_lung_mask_2d[r_mask] = 0
                
    old_binary = new_lung_mask_2d>0
    old_mask = new_lung_mask_2d
    new_binary = binary_closing(old_binary,disk(int(5*(w/512)**2))) # fill in eroded gaps, adjust given image dimension
    new_binary = remove_small_objects(remove_small_holes(new_binary, area_threshold=int(TRACHEA_REMOVAL_AREA*(w/512)**2)), 
                                min_size = int(TRACHEA_REMOVAL_AREA*(w/512)**2)) # need to fill up and then remove sizes that are smaller than trachea_removal_area, adjust by size because the trachea area usually decreaseswith area dimension
    if np.max(old_mask) == 1:
        return new_binary
    '''
    preserve labels of the original ternary mask
    '''
    new_lung_mask_2d = new_binary.copy()
    label_old_binary = label(old_binary)
    label_new_binary = label(new_binary)
    old_mask_dict = {i:label_old_binary==i for i in range(1, np.max(label_old_binary)+1)}
    new_mask_dict = {j:label_new_binary==j for j in range(1, np.max(label_new_binary)+1)}
    for j in range(1, np.max(label_new_binary)):
        new_mask_j = new_mask_dict[j]
        overlaps = {i:np.sum(np.logical_and(old_mask_dict[i], new_mask_j)) for i in old_mask_dict}
        best_i = max(overlaps, key = lambda k: overlaps[k])
        new_lung_mask_2d[new_mask_j] = old_mask[old_mask_dict[best_i]][0] # preserve labels
    return new_lung_mask_2d
    
        
def get_affine_transform(mask_2d):
    
    '''
    sometime the lung ct scan is tilted so need to be rotated properly to make the lung bottom level
    
    rotation is defined by using the line between the lowest points of each lung as baseline and aim to transform this line
    to be exactly horizontal
    
    tranlation is defined by placing the rotated img as an incribed square/rectangle in a new square/reactangle so
    that the original img center is still at the img center

    INPUT: an axial mask with one lung (the left in the img) labeled as 2 and the other labeled as 1
    OUTPUT: the rotated axial_mask
    '''
    
    first_xs, first_ys = np.where(mask_2d==2)
    second_xs, second_ys = np.where(mask_2d==1)

    x1 = max(first_xs) # lowest point for lung region 1 on 2d image
    y1 = min([first_ys[k] for k in range(len(first_ys)) if first_xs[k]==x1]) # get leftmost lowest point
    x2 = max(second_xs) # lowest point for lung region 2 on 2d image
    y2 = max([second_ys[k] for k in range(len(second_ys)) if second_xs[k]==x2]) # get rightmost lowest point
    
    r = math.atan(-((x2-x1)/(y2-y1))) 
    # proper counterclockwise rotation angle by connecting the line defined by the two lowest points 
    # want the line to become horizontal for proer segmentation
    
    h,w=mask_2d.shape
    h2 = int(w*math.sin(abs(r))+h*math.cos(abs(r))) 
    #draw a graph to visualize that this is correct if we want the rotated img to be inscribed in the new enlarged img
    w2 = int(h*math.sin(abs(r))+w*math.cos(abs(r)))
    #print(x1,y1,x2,y2,r,h2,w2)

    old_c1 = h/np.sqrt(2)*math.sin(math.pi/4-r)
    #old_center coords (easier to see if draw a picture, positioning the old img at the ccenter of the new enlarged img
    old_c2 = h/np.sqrt(2)*math.cos(math.pi/4-r) 

    new_c1, new_c2 = h2/2, w2/2 # new center coords

    dc1,dc2 = new_c1-old_c1, new_c2-old_c2 # translate center properly at the middle of the img
    tform = AffineTransform(rotation=r, translation=(dc1,dc2)) 
    return r,(h2,w2),tform

def get_3d_segmented_lung(lung_ct, return_mask=False, fn=None): # use axial
    '''
    INPUT (lung_ct): original 3d lung ct scan
    PIPELINE:
    1. get a ternary mask of one of the middle slices
    2. use the mask to get affine transform
    3. apply affine transform to each slice so that each axial scan is horizontal
    4. segment out the mask on each slice
    5. postprocess remove airways
    6. rotate the mask back and compile the masks into a 3D array
    OUTPUT: two 3d ternary lung mask, the first without airway removed, the second with airway removed
    '''
    segmented_lung_ct = np.zeros(lung_ct.shape)
    segmented_lung_ct_airway_removed = np.zeros(lung_ct.shape)
    d,h,w = lung_ct.shape
    index = d // 2
    binarized_ct = np.zeros((d,h,w))
    for i in range(d):
        binarized_ct[i,...] = remove_borders(lung_ct[i,...], very_superior_end=i<d/10) # remove borders and binarize lung_ct
        
    sample_mask=fn(binarized_ct[index], 
                        return_mask=return_mask,
                        inferior_end = index > d*3//4)
    sample_mask=separate_lungs_2d(sample_mask)

    while len(np.unique(sample_mask)) <= 2: # need a mask that has both 1 and 2
        if index < 0:
            print('cannot separate lungs properly')
            return None
            break
        index -= 1
        sample_mask=fn(binarized_ct[index], 
                        return_mask=return_mask,
                        inferior_end = index > d*3//4,
                        superior_end = index < d//4)
        sample_mask=separate_lungs_2d(sample_mask)
   
    
    r,out_shape,tform = get_affine_transform(sample_mask)
    area_ratio = np.sum(sample_mask == 2)/np.sum(sample_mask == 1) 
    if area_ratio > 5 or area_ratio < 1/5: # if area difference is too large, don't bother roatet it
        r = 0
    cval = np.min(lung_ct)
    if abs(r)/math.pi <= ANGLE_THRES:# if tilted degree is less than 5 degree, then don't bother rotate it
        for i in range(d): 
            segmented_lung_ct[i,...]=fn(binarized_ct[i], 
                            return_mask=return_mask,
                            inferior_end = i > d*3//4,
                            superior_end = index < d//4)
            segmented_lung_ct[i,...]=separate_lungs_2d(segmented_lung_ct[i,...])
            segmented_lung_ct_airway_removed[i,...]=segmented_lung_ct[i,...].copy()
            segmented_lung_ct_airway_removed[i,...]=postprocess_remove_airway(lung_ct[i,...], segmented_lung_ct_airway_removed[i,...])
            segmented_lung_ct[i,...]=separate_lungs_2d(segmented_lung_ct[i,...])
        new_mask = separate_into_two_lungs(segmented_lung_ct)
        new_mask_airway_removed = separate_into_two_lungs(segmented_lung_ct_airway_removed)

        
    else:
        rotated_segmented_lung_ct = np.zeros((d, out_shape[0],out_shape[1]))

        for i in range(d): 
            rotated_img = warp(lung_ct[i], tform.inverse, output_shape=out_shape, 
                                        preserve_range=True, mode='constant',
                                        cval=cval)
            rotated_binary = warp(binarized_ct[i], tform.inverse, output_shape=out_shape, 
                                        preserve_range=True, mode='constant',
                                        cval=0)
            rotated_segmented_lung_ct[i,...] = fn(rotated_binary, 
                            return_mask=return_mask,
                            inferior_end = i > d*3//4,
                            superior_end = index < d//4)
            rotated_segmented_lung_ct[i,...]=separate_lungs_2d(rotated_segmented_lung_ct[i,...])
        
        rotated_segmented_lung_ct = separate_into_two_lungs(rotated_segmented_lung_ct)
            
        for i in range(d):                    
            mask = warp(rotated_segmented_lung_ct[i,...], tform, output_shape=(h,w), 
                                                 preserve_range=True, mode='constant',
                                                 cval=0)
            mask_airway_removed = mask.copy()
            mask_airway_removed = postprocess_remove_airway(lung_ct[i,...], mask_airway_removed)
            segmented_lung_ct[i,...]=mask
            segmented_lung_ct_airway_removed[i,...]=mask_airway_removed
        new_mask = segmented_lung_ct
        new_mask_airway_removed = segmented_lung_ct_airway_removed
    
    return new_mask, new_mask_airway_removed

def refine_lung_segmentation(seg_lung_axial_mask, depth=10):
    '''
    take the intersection of the neighboring `depth` mm of axial masks to assign to that slice
    act to remove small regions that is not consistently appearing in the z direction

    INPUT: 3d lung mask
    OUTPUT: 3d lung mask
    '''
    shape = seg_lung_axial_mask.shape
    refined_seg_mask = seg_lung_axial_mask.copy()
    

    for i in range(0, shape[0]):
        label_image = label(seg_lung_axial_mask[i]>0)
        for ri in range(1,np.max(label_image)+1):
            ri_mask = label_image==ri
            ri_mask_area = np.sum(ri_mask)
            overlap = []
            for j in range(max(0,i-depth//2), min(max(0,i-depth//2)+depth, shape[0])): 
                overlap.append(np.sum(seg_lung_axial_mask[j][ri_mask]>0)>=ri_mask_area/2)
            if np.sum(overlap) <= min(depth//2, (shape[0]-i)//2):
                refined_seg_mask[i][ri_mask] = 0
    
    for i in range(shape[0]*3//4+1, shape[0]):
        '''
        # find the point where the slice before and the slice after has no intersection at all
        # which means that the slice before is lung and the slice after is trachea/other fragment
        '''
        if np.sum((seg_lung_axial_mask[i-1]>0) & (seg_lung_axial_mask[i]>0)) == 0:
            refined_seg_mask[i:,...] = 0
            break
        
    return refined_seg_mask

def remove_airway_borders(original_mask): # this does not work if the border thickness is greater than 1
    '''
    remove borders of airways that did not get removed by postprocess_remove_airway_2d
    INPUT: 2d mask
    OUTPUT: 2d mask
    '''
    new_mask = original_mask.copy().astype('int')
    
    def remove_hollow_circles(mask_2d):
        regions = regionprops(label(mask_2d>0, connectivity=1))
        regions.sort(key=lambda r:r.area)
        if len(regions) == 0:
            return mask_2d
        i = 1
        while i < len(regions) and regions[-i-1].area > regions[-i].area / 5:
            i += 1
        
        for r in regions[:-i]:
            for c in r.coords:
                mask_2d[c[0],c[1]] = 0
        return mask_2d
    
    for i in range(new_mask.shape[0] // 2): # just the first half slices require this operation
        for _ in range(2): # 2 rounds of keeping only 1-connected voxels, can remove border of airwat
            new_mask[i] = remove_hollow_circles(new_mask[i])
        
    return new_mask

    
def separate_into_two_lungs(original_mask):
    '''
    INPUT: 3d mask
    OUTPUT: a refined ternary 3d mask where each slice is separated into left and right lungs
    '''
    new_mask = original_mask.copy().astype('int')
    d,h,w = new_mask.shape
    
    def get_height_at_each_width_new(region_mask): # this is different from the same named function named outside `get_segmented_lungs_trial_and_error
        '''
        # this is different from the same named function named inside `get_segmented_lungs_trial_and_error
        # this function is equivalent to drawing a vertical line at a certain y value, and get the difference between x1 and x2
        # x1 is when the line first intersects the region, x2 is when the line intersects the region for the second time
        '''        
        xs, ys = np.where(region_mask > 0)
        height_at_each_width = []
        for j in range(region_mask.shape[1]):
            if j not in ys:
                height_at_each_width.append(0)
            else:
                indices = np.where(ys==j)[0]
                height_at_each_width.append(max(xs[indices])-min(xs[indices])+1)
        return height_at_each_width

    for i in range(d//10, d): # usually just the middle slices require this operation
        xs, ys = np.where(new_mask[i] > 0)
        if (np.sum(new_mask[i]==1) < 20 or np.sum(new_mask[i]==2) < 20) and not np.all(ys < w/2) and not np.all(ys >= w/2):
        # one label, with segmentation in both halves of the image, and total number of areas is <= 2
            xs, ys = np.where(new_mask[i] > 0)
            height_at_each_width = get_height_at_each_width_new(new_mask[i])

            start1 = 0
            while start1 < w and height_at_each_width[start1] == 0:
                start1+=1
            # start1 is the first y coord when the left region has voxels 
            if start1 == w:
                continue
            # end1 is the last y coord before when the left region has no voxels
            end1 = start1
            while end1 < w and height_at_each_width[end1] > 0:
                end1+=1
            start2 = end1
            while start2 < w and height_at_each_width[start2] == 0:
                start2+=1
            # start2 is the starting point for the right region
            if start2 == w:
                '''
                this mean that there is only one connected region
                we use the y coord where the height of the mask at this y values is the smallest as separateing point
                '''
                diff = (end1-start1)
                new_start = start1+diff//4
                new_end = end1-diff//4

                min_height_widths = [j for j in range(new_start, new_end) if height_at_each_width[j] == min(height_at_each_width[new_start:new_end])]
                # using middle half to avoid selecting border of lungs as sep
                min_height_widths.sort(key=lambda j: abs(j-w/2)) # get j that is closest to the center as the base for separation
                sep = min_height_widths[0]
            else:
                sep = (end1+start2)/2
            
            left_xs, left_ys = xs[ys <= sep], ys[ys <= sep]
            for k in range(len(left_xs)):
                new_mask[i,left_xs[k], left_ys[k]] = 2
            right_xs, right_ys = xs[ys > sep], ys[ys > sep]
            for k in range(len(right_xs)):
                new_mask[i,right_xs[k], right_ys[k]] = 1
    
    return new_mask
    

def get_resized_lung_ct(image_dir):
    '''
    INPUT: a string in the form of 'parentDir/subDir/.../imageDir' or 
            '['parentDir1/subDir1/.../imageDir1', 'parentDir2/subDir2/.../imageDir2',...]' 
            (i.e. string of the list of multiple directories)
    OUTPUT: windowed lung ct resized to 1mm voxels
    '''
    
    if '[' in image_dir and ',' in image_dir and ']' in image_dir: # list stored as string
        image_dir = image_dir.strip("]['").split("',")[0] 
        # due to how csv stores the list, have to do it this way
    ct_slices = [pydicom.dcmread(file) for file in glob.glob(image_dir + '/*')]
    try:
        full_ct = get_3d_image(ct_slices, windowing = False)
    except ValueError as e:
        print(e)
        return None
    spacings = get_spacings(ct_slices[0])
    window_center, window_width, intercept, slope = get_windowing(ct_slices[0])
    full_ct_resized, new_spacings = resize_image(full_ct, spacings)                   
    lung_ct = window_image(full_ct_resized, -600, 1500, intercept, slope)
    return lung_ct
    
    

        
    

def create_thresholding_3d_faster(img, mask, patch_size=1):
    '''
    INPUT: img: 3d lung ct; mask: 3d lung mask; patch_size: length of 3d kernel
    OUTPUT: 3d patch average 
    '''
    if patch_size == 1:
        new_img = img.copy()
        new_img[mask==0] = 0
        return new_img
    dist = patch_size
    mask = (mask > 0).astype('int') # convert to binary
    temp_vals = np.multiply(img, mask)
    kernel = np.ones((dist, dist, dist))
    temp_vals_sum_by_cube = ndi.correlate(temp_vals, kernel, mode='constant', cval=0.0)
    mask_sum_by_cube = ndi.correlate(mask, kernel, mode='constant', cval=0.0)
    mask_sum_by_cube[temp_vals_sum_by_cube==0] = 1 # avoid division error
    return temp_vals_sum_by_cube / mask_sum_by_cube 

def create_thresholding_2d_faster(img, mask, patch_size=1):
    '''
    INPUT: img: 3d lung ct; mask: 3d lung mask; patch_size: length of 2d kernel
    OUTPUT: 3d patch average 
    '''
    if patch_size == 1:
        new_img = img.copy()
        new_img[mask==0] = 0
        return new_img
    dist = patch_size
    patched_avgs = img.copy()
    kernel = np.ones((dist, dist))/dist**2

    for i in range(img.shape[0]):
        temp_vals = np.multiply(img[i], mask[i])
        temp_vals_sum_by_square = ndi.correlate(temp_vals, kernel, mode='constant', cval=0.0)
        mask_sum_by_square = ndi.correlate(mask[i], kernel, mode='constant', cval=0.0)
        mask_sum_by_square[temp_vals_sum_by_square==0] = 1 # avoid division error
        patched_avgs[i,...] = temp_vals_sum_by_square / mask_sum_by_square 
    
    return patched_avgs




def get_ternarymask_simplified(lung_ct):
        
    tup = get_3d_segmented_lung(lung_ct, fn=get_segmented_lungs_trial_and_error,
                                               return_mask=True)
    if tup is None:
        return None
    axial_mask, axial_mask_airway_removed = tup

    axial_mask_airway_removed = refine_lung_segmentation(axial_mask_airway_removed)
    axial_mask_airway_removed = separate_into_two_lungs(axial_mask_airway_removed)
    
    return axial_mask_airway_removed
    


def wrapper_fn(acc_num, DICOM_dir, npz_save_dir, thres=-950):
    
    lung_ct = get_resized_lung_ct(DICOM_dir)
    if lung_ct is None:
        return None
    mask = get_ternarymask_simplified(lung_ct)
    if mask is None:
        return None
    
    volume = np.sum(mask>0)
    binary_mask = mask > 0

    scores = {'accession number': acc_num}
    for ps in [1,3,5]:
        patched_avgs_3D = create_thresholding_3d_faster(lung_ct, binary_mask, patch_size=ps)
        score_3D = np.sum(patched_avgs_3D[binary_mask] < thres)/volume
        scores['3D ps='+str(ps)]=score_3D
    for ps in [3,5,7]:
        patched_avgs_2D = create_thresholding_2d_faster(lung_ct, binary_mask, patch_size=ps)
        score_2D = np.sum(patched_avgs_2D[binary_mask] < thres)/volume
        scores['2D ps='+str(ps)]=score_2D
    
    print(scores)
    np.savez(npz_save_dir+str(acc_num),lung_ct=lung_ct, lung_mask=mask, scores=scores) 
    return scores
    


def run_parallel_UCSF(i, acc_nums, Filepaths, npz_save_dir):
    return wrapper_fn(acc_nums[i], Filepaths[i], npz_save_dir)
