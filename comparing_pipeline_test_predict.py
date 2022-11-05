# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:15:37 2022

@author: binda
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:52:15 2022

@author: binda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:48:54 2022

@author: binda
"""

# step 1: import libraries
from utils \
    import get_image_gdf_in_directory, geotif_to_png
from azimuth_try import azimuth, area
from module_placement import module_placement_options
from mask_generation import import_vector_labels
from utils import prediction_raster_to_vector, get_progress_string, prediction_raster_to_vector1
from definitions import LABEL_CLASSES_SUPERSTRUCTURES, LABEL_CLASSES_SEGMENTS, IMAGE_SHAPE, \
    FILE_VECTOR_LABELS_SEGMENTS, FILE_VECTOR_LABELS_SUPERSTRUCTURES, DIR_IMAGES_PNG, \
    df_technical_potential_LUT, EPSG_METRIC, DIR_PREDICTIONS, LABEL_CLASSES_6, LABEL_CLASSES_PV_AREAS, LABEL_CLASSES_ROOFLINE, OUTPUT_FROM_PREDICTION, \
    DIR_IMAGES_GEOTIFF_TRASH
from definitions import \
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,\
    FILE_VECTOR_LABELS_SEGMENTS,\
    FILE_VECTOR_LABELS_PV_AREAS,\
    DIR_BASE, \
    DIR_DATA, \
    DIR_ROOFL_GABLE, \
    OUTPUT_AUGMENT_MASK_ROOFLINE, \
    DIR_IMAGES_GEOTIFF_COPY, \
    DIR_PREDICTED_IMAGES, \
    PNG_TO_GEOTIFF, PNG_TO_GEOTIFF1, COMPARISON_SEGMENT, COMPARISON_GUTTER, COMPARISON_ORIG, LABEL_CLASSES_SEGMENTS
    
#from gold_gutter import segment_generation
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
from utils import get_wartenberg_boundary, get_static_map_bounds, save_as_geotif
import geopandas as gpd
import cv2
import pickle
import matplotlib.colors as colors
import shapely
import itertools
shapely.speedups.disable()
from gold_copy import segment_generation
#from visualization import visualize_module_placement, box_plot_E_gen_TUM_CI

import cv2
import numpy as np
from PIL import Image
"""
def remove_bg(input_file, output_file):
    if not os.path.isdir(output_file):
        os.mkdir(output_file)
    input_files = [geotif[:-4] for geotif in os.listdir(input_file) if geotif[-4:]=='.png']
    output_files = [png[:-4] for png in os.listdir(output_file) if png[-4:]=='.png']
    missing_pngs_list = [geotif for geotif in input_files if geotif not in output_file]
    
    for i, img in enumerate(missing_pngs_list):
        input_image = os.path.join(input_file, img + '.png')
        output_path = os.path.join(output_file, img + '.png')
        roof_read = cv2.imread(input_image,1)
        
       
        
        img_gray = cv2.cvtColor(roof_read, cv2.COLOR_BGR2GRAY)
        img_gray[np.where(img_gray == 1)] = 0
        img_gray.flatten()
        img_gray[img_gray != 0] = 255
        img_gray_255 = img_gray/255
        img_with_border = cv2.copyMakeBorder(img_gray_255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        resize_img = cv2.resize(img_with_border, (512,512))
        cv2.imwrite(output_path, resize_img)
       
    return
    
remove_bg(OUTPUT_FROM_PREDICTION, DIR_PREDICTED_IMAGES)    
    

"""
"""
def create_geotiff(input, image, output):
    if not os.path.isdir(output):
        os.mkdir(output)

    input_files = [geotif[:-4]
                   for geotif in os.listdir(input) if geotif[-4:] == '.png']

    image_files = [png[:-4] for png in os.listdir(image) if png[-4:] == '.png']
    # print(image_files)
    output_files = [png[:-4]
                    for png in os.listdir(output) if png[-4:] == '.png']
    missing_pngs_list = [
        geotif for geotif in input_files if geotif not in output]

    for i, img in enumerate(missing_pngs_list):

        geotiff_image = os.path.join(input, img + '.png')
        predict_image = os.path.join(image, img + '.png')
        output_path = os.path.join(output, img + '.png')

        raster_src = gdal.Open(geotiff_image, gdal.GA_ReadOnly)
        # coordinates of upper left corner and resolution
        ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()
        # coordinates of lower right corner
        lrx = ulx + (raster_src.RasterXSize * xres)
        lry = uly + (raster_src.RasterYSize * yres)
        #image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
        bbox_gen = [uly, lrx, lry, ulx]
        image_gen = cv2.imread(predict_image, 1)
        #image = cv2.imwrite(output_path, image_read)
        # print(image)

        save_as_geotif(bbox_gen, image_gen, output_path)

    return


created_geotiff = create_geotiff(
    DIR_IMAGES_GEOTIFF_TRASH,
    DIR_PREDICTED_IMAGES,
    PNG_TO_GEOTIFF
)

 """   
 #change COMPARISON_ORIG to COMPARISON_SEGMENT and LABEL_CLASSES_SEGMENTS to LABEL_CLASSES_PV_AREAS
 #prediction_raster_to_vector1 to prediction_raster_to_vector
def orientation():
    prediction_mask_filename = os.listdir(COMPARISON_ORIG)
    prediction_mask_filepath = [os.path.join(
        COMPARISON_ORIG, file) for file in prediction_mask_filename]
    prediction_mask = [cv2.imread(prediction, 0)
                       for prediction in prediction_mask_filepath]
    
    prediction_mask_filename1 = os.listdir(COMPARISON_GUTTER)
    prediction_mask_filepath1 = [os.path.join(
        COMPARISON_GUTTER, file) for file in prediction_mask_filename1]
    prediction_mask1 = [cv2.imread(prediction, 0)
                       for prediction in prediction_mask_filepath1]
    



    with open("data\\gdf_image_boundaries.pkl", 'rb') as f:
        gdf_images = pickle.load(f)
    gdf_images.id = gdf_images.id.astype(int)

    gdf_predictions = []
    dir_lists = []
    
    

    # # c) convert raster data of superstructures to vector data
    for i, (mask, mask1) in enumerate(zip(prediction_mask, prediction_mask1)):
        mask_id = prediction_mask_filename[i][:-4]
        gdf_image = gdf_images[gdf_images.id == int(mask_id)]
        image_bbox = gdf_image.geometry.iloc[0]
        gdf_predictions = prediction_raster_to_vector1(
            mask, mask_id, image_bbox, LABEL_CLASSES_ROOFLINE, IMAGE_SHAPE)
        gdf_predictions = gdf_predictions.iloc[2: , :]
        gdf_predictions.reset_index(drop=True, inplace=True)
        gdf_labels = gpd.GeoDataFrame(gdf_predictions, geometry='geometry')
        gdf_labels.crs = 4326
        
        
        mask_id1 = prediction_mask_filename1[i][:-4]
        gdf_image1 = gdf_images[gdf_images.id == int(mask_id1)]
        image_bbox1 = gdf_image1.geometry.iloc[0]
        gdf_predictions1 = prediction_raster_to_vector(
            mask1, mask_id1, image_bbox1, LABEL_CLASSES_ROOFLINE, IMAGE_SHAPE)
        gdf_predictions1 = gdf_predictions1[:-1]
        gdf_predictions1.reset_index(drop=True, inplace=True)
        
        gdf_labels1 = gpd.GeoDataFrame(gdf_predictions1, geometry='geometry')
        gdf_labels1.crs = 4326
        #min_rot_each = gdf_labels.minimum_rotated_rectangle
        save_directory1 = "C:\\Users\\binda\\Downloads\\check\\comparison_gutter_big"+ "\\" + str(mask_id1) + ".shp"
        gdf_labels1.to_file(save_directory1)
        
        dir_list = segment_generation(gdf_labels, gdf_labels1, mask_id1)
        dir_lists.append(dir_list)
        gdf_labels['direction']= dir_list
        #gdf_labels['direction'] = [dir_list]*len(gdf_labels)
        save_directory = "C:\\Users\\binda\\Downloads\\check\\comparison_segment"+ "\\" + str(mask_id) + ".shp"
        gdf_labels.to_file(save_directory)
       
    print(dir_lists)

    with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list2.txt", "w") as output:
        output.write(str(dir_lists))

    dir_lists
    merged = list(itertools.chain.from_iterable(dir_lists))
                
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

      
       
       
       
    return


orientation()



