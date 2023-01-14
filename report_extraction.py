#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd

def get_emphysema_data(text):
    text = text.replace(r'\n', ' ')
    #print('emphysema' in text)
    if 'emphysema' in text:
        return text
    else:
        return 'none'

c = -1
def get_emphysema_extent(emphysema_text):
    global c 
    c+=1
    if emphysema_text == 'none':
        return 'none'
    else:
        emphysema_text = emphysema_text.lower()
        extent_levels = ['no','mild', 'moderate', 'severe']
        middlezone_levels = [r'mild[\s-]to[\s-]moderate', r'moderate[\s-]to[\s-]severe']
        extents = []
        for extent in middlezone_levels:
            emphysema_extent = re.findall(extent+'\s[a-z\s]*'+'emphysema', emphysema_text)
            if len(emphysema_extent) > 0:
                extents.append(extent)
        if len(extents) > 1:
            texts = emphysema_text.split(' ')
            print(emphysema_text)
            print(c)
            return extents
        elif len(extents) > 0:
            return extents[0]
        for extent in extent_levels:
            emphysema_extent = re.findall(extent+'\s[a-z\s]*'+'emphysema', emphysema_text)
            if len(emphysema_extent) > 0:
                extents.append(extent)
        if 'no' in extents:
            print(emphysema_text)
            print(c)
        if len(extents) > 1:
            texts = emphysema_text.split(' ')
            print(emphysema_text)
            print(c)
            return extents
        elif len(extents) > 0:
            return extents[0]
        return 'Not specified' # not specified means the extent of emphysema is not recorded but there is emphysema

# read your patient metadata file, it should contain a colu n called `Report Text`
patient_df = ...
emphysema_info = patient_df['Report Text'].map(get_emphysema_data)
emphysema_in_report = patient_df['Report Text'].map(lambda x: 'emphysema' in x.lower())
emphysema_extent_info = emphysema_info.map(get_emphysema_extent)

# inspect the printed c and full report text to make manual adjustment for emphysema extents

emphysema_extent_info[7] = 'moderate'
emphysema_extent_info[87] = 'moderate'
emphysema_extent_info[148]= 'none'
emphysema_extent_info.loc[emphysema_extent_info=='mild[\s-]to[\s-]moderate'] = 'mild to moderate'
emphysema_extent_info.loc[emphysema_extent_info=='moderate[\s-]to[\s-]severe'] = 'moderate to severe'

# store your emphysema extent with their corresponding accession number somewhere