# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:39:36 2022

@author: binda
"""
from mask_generation \
    import vector_labels_to_masks, import_vector_labels
    
from definitions \
    import FILE_VECTOR_LABELS_SEGMENTS, DIR_MASKS_SEGMENTS
    
from utils \
    import get_image_gdf_in_directory, geotif_to_png, prediction_raster_to_vector
import pickle
import os
import cv2
import numpy as np
import geopandas as gpd
import itertools
LABEL_CLASSES_SEGMENTS = 'C:\\Users\\binda\\Downloads\\try_azimuth\\segments_reviewed.csv'
DIR_IMAGES_GEOTIFF = 'C:\\Users\\binda\\Downloads\\geotiff_try\\test'

gdf_images = get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF)

"""
gdf_labels_segments = vector_labels_to_masks(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS,
    'segments',
    LABEL_CLASSES_SEGMENTS,
    gdf_images,
    filter=False
)
"""

prediction_mask_filename = os.listdir(DIR_IMAGES_GEOTIFF)
prediction_mask_filepath = [os.path.join(
    DIR_IMAGES_GEOTIFF, file) for file in prediction_mask_filename]
prediction_mask = [cv2.imread(prediction, 0)
                   for prediction in prediction_mask_filepath]

with open("data\\gdf_image_boundaries.pkl", 'rb') as f:
    gdf_images = pickle.load(f)
gdf_images.id = gdf_images.id.astype(int)

gdf_predictions = []
dir_lists = []
dir_lists1 = []
azimuth = []
IMAGE_SHAPE = cv2.imread(DIR_IMAGES_GEOTIFF + '\\' + os.listdir(DIR_IMAGES_GEOTIFF)[0], 0).shape
label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

LABEL_CLASSES_PV_AREAS = dict(zip(np.arange(0, len(label_classes_segments_18)), label_classes_segments_18))

for i, (mask) in enumerate(prediction_mask):
    mask_id = prediction_mask_filename[i][:-4]
    gdf_image = gdf_images[gdf_images.id == int(mask_id)]
    image_bbox = gdf_image.geometry.iloc[0]
    gdf_predictions = prediction_raster_to_vector(
        mask, mask_id, image_bbox, LABEL_CLASSES_PV_AREAS, IMAGE_SHAPE)
    #gdf_predictions = gdf_predictions.iloc[2: , :]
    gdf_predictions.reset_index(drop=True, inplace=True)
    try:
        azimuth_value = gdf_predictions['label_type'].tolist()
    except:
        pass
    azimuth.append(azimuth_value)
    gdf_labels = gpd.GeoDataFrame(gdf_predictions, geometry='geometry')
    gdf_labels.crs = 4326
    try: 
        save_directory1 = "C:\\Users\\binda\\Downloads\\try_azimuth"+ "\\"+str(mask_id)+ ".shp"
        gdf_labels.to_file(save_directory1)
    except:
        pass

merged = list(itertools.chain.from_iterable(azimuth))
with open("C:\\Users\\binda\\Downloads\\try_azimuth\\list1.txt", "w") as output:
        output.write(str(merged))
with open("C:\\Users\\binda\\Downloads\\try_azimuth\\list2.txt", "w") as output:
        output.write(str(azimuth))
        
from collections import Counter

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]
    

print(Most_Common(merged))


import collections
        # intializing the arr
arr = merged
        # getting the elements frequencies using Counter class
elements_count = collections.Counter(arr)
        # printing the element and the frequency
for key, value in elements_count.items():
   print(f"{key}: {value}")
