# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:51:53 2022

@author: binda
"""
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
shapely.speedups.disable()

from definitions import \
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,\
    FILE_VECTOR_LABELS_SEGMENTS,\
    FILE_VECTOR_LABELS_PV_AREAS,\
    DIR_BASE, \
    DIR_DATA, \
    DIR_ROOFL_GABLE, \
    OUTPUT_AUGMENT_MASK_ROOFLINE, \
    DIR_IMAGES_GEOTIFF
from definitions import LABEL_CLASSES_SUPERSTRUCTURES, LABEL_CLASSES_SEGMENTS, IMAGE_SHAPE, \
    FILE_VECTOR_LABELS_SEGMENTS, FILE_VECTOR_LABELS_SUPERSTRUCTURES, DIR_IMAGES_PNG, \
    df_technical_potential_LUT, EPSG_METRIC, DIR_PREDICTIONS, LABEL_CLASSES_6, LABEL_CLASSES_PV_AREAS, LABEL_CLASSES_ROOFLINE
from utils import prediction_raster_to_vector, get_progress_string
from mask_generation import import_vector_labels
from module_placement import module_placement_options
#from visualization import visualize_module_placement, box_plot_E_gen_TUM_CI    
from azimuth_try import azimuth, area
        
    #
ge_img_filepath = "D:\\RID-master\\RID-master\\data\\images_roof_centered_geotiff"
image_path = "D:\\RID-master\\RID-master\\venv\\data\\augment_mask_output_roofline"
save_path =  "D:\\RID-master\\RID-master\\png_to_geotiff\\png"  
 
from utils \
    import get_image_gdf_in_directory, geotif_to_png


def create_geotiff(input,image, output):
    if not os.path.isdir(output):
        os.mkdir(output)
        
    input_files = [geotif[:-4] for geotif in os.listdir(input) if geotif[-4:] == '.tif']
    
    image_files = [png[:-4] for png in os.listdir(image) if png[-4:] == '.png']
    #print(image_files)
    output_files = [png[:-4] for png in os.listdir(output) if png[-4:] == '.tif']
    missing_pngs_list = [geotif for geotif in input_files if geotif not in output]
        
        
    for i, img in enumerate(missing_pngs_list):
            
        ge_img_filepath = os.path.join(input, img + '.tif')
        image_path_1 = os.path.join(image, img + '.png')
        output_path = os.path.join(output, img + '.tif')
        
        raster_src = gdal.Open(ge_img_filepath, gdal.GA_ReadOnly)
        ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
        lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
        lry = uly + (raster_src.RasterYSize * yres)
            #image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
        bbox_gen = [uly, lrx, lry, ulx]
        image_gen = cv2.imread(image_path_1, 1)
        #image = cv2.imwrite(output_path, image_read)
        #print(image)
        
            
        save_as_geotif(bbox_gen, image_gen, output_path)
        
    return
        
        
#create_geotiff(
   #ge_img_filepath,
   #image_path,
   #save_path
   #)

##note: the images created in geotiff_images is not yet our output images. Work need to be done
##hence commented out. 
##hence we are using our own created image for our better understanding

def pv_potential_analysis():
    # a) load superstructure predictions
    prediction_mask_filenames = os.listdir(DIR_PREDICTIONS)
    prediction_mask_filepaths = [os.path.join(DIR_PREDICTIONS, file) for file in prediction_mask_filenames]
    prediction_masks = [cv2.imread(prediction, 0) for prediction in prediction_mask_filepaths]

    # b) load superstructure ground truth
    gdf_superstructures_GT = import_vector_labels(
        FILE_VECTOR_LABELS_SUPERSTRUCTURES,
        'superstructures',
        LABEL_CLASSES_SUPERSTRUCTURES
    )
    # rename class colomn
    gdf_superstructures_GT = gdf_superstructures_GT.rename(columns={'class_type': 'label_type'})

    # transform coordinate system
    gdf_superstructures_GT.crs = 4327
    
    gdf_superstructures_GT = gdf_superstructures_GT.to_crs(EPSG_METRIC)
   
    with open("data\\gdf_image_boundaries.pkl", 'rb') as f:
        gdf_images = pickle.load(f)
    gdf_images.id = gdf_images.id.astype(int)

    gdf_superstructures_PR = gpd.GeoDataFrame({'mask_id': [], 'label_type': [], 'geometry': []})
    gdf_predictions = []

    # # c) convert raster data of superstructures to vector data
    for i, mask in enumerate(prediction_masks):
        mask_id = prediction_mask_filenames[i][:-4]
        gdf_image = gdf_images[gdf_images.id == int(mask_id)]
        image_bbox = gdf_image.geometry.iloc[0]
        gdf_predictions = prediction_raster_to_vector(mask, mask_id, image_bbox, LABEL_CLASSES_ROOFLINE, IMAGE_SHAPE)
        #when only one class in used as input, the roofs which touch the outerline would be white
        #print(gdf_predictions)
        #gdf_predictions.plot(color='white', edgecolor='black')
        #gdf_predictions.to_csv("D:\\RID-master\\RID-master\\venv\\shapefile\\countries.shp")
        #gdf_predictions.append(gdf_predictions)
    #return gdf_predictions
        
        gdf_labels = gpd.GeoDataFrame(gdf_predictions, geometry='geometry')
        #gdf_labels
        gdf_labels.crs=4326              
        #note: add cmap=colors.ListedColormap(list(color_dict.values())) inside plot() for color_dict


##todo: what I try to do
        #mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
        mrrs = gdf_labels.geometry.apply(lambda geom: geom)
        mrrs
        gdf_labels.crs=4326
        mrrs_area = gdf_labels.geometry.area
        gdf_labels['az'] = mrrs.apply(azimuth)
        gdf_labels['area'] = mrrs_area
        
        gdf_labels.loc[gdf_labels['az'] == '90.00', 'az'] = -10
        #gdf_labels['az'].mask(gdf_labels['az'] == '90.00', 0, inplace=True)
        gdf_labels
        #background = np.where(np.array(b_label) == 0, 0, 1)
        
        #for region in gdf_labels.az:
            #row = gdf_labels[gdf_labels.az==region]
            #row.crs = 4326
            #area_list = np.round(np.array(row.geometry.area), 2)
            #print(area_list)
        

        ax = gdf_labels.plot('az', aspect =1, legend=True, edgecolor='black')
      
        plt.axis('off')
        plt.grid(b=None)
        #ax.figure.savefig('temp.png',aspect='normal', dpi=512)
        
        
        #b= mrrs.boundary.plot(ax=ax, alpha=0.5)
    return


pv_potential_analysis()
    
###trying another method






######################## just checking
def open_geotiff(file_path):
    """
        Function to open a GEOTIFF

        Inputs
        ----------
        file_path : string
            string to GEOTIFF file path

        Outputs
        ----------
        image : numpy array
            RGB image as a numpy array with shape [x_pixels, y_pixels, 3]

        image_bbox : list
            bounding box coordinates of image: [x_min, y_min, x_max, y_max]

        coordinate_system : string
            string with name of coordinate system of coordinates in image_bbox

        """

    # load  image
    raster_src = gdal.Open(file_path, gdal.GA_ReadOnly)

    # get image data and rearrange to get a numpy array with RGB image shape
    data = raster_src.ReadAsArray()
    image = np.dstack((data[0, :, :], data[1, :, :], data[2, :, :]))

    # get image bounding box from geotiff
    ulx, xres, xskew, uly, yskew, yres = raster_src.GetGeoTransform()  # coordinates of upper left corner and resolution
    lrx = ulx + (raster_src.RasterXSize * xres)  # coordinates of lower right corner
    lry = uly + (raster_src.RasterYSize * yres)  # coordinates of lower right corner
    image_bbox = [ulx, lry, lrx, uly]

    # get string of spatial reference system
    src = osr.SpatialReference()
    src.ImportFromWkt(raster_src.GetProjection())
    coordinate_system = (src.GetAttrValue('geogcs'))

    return image, image_bbox, coordinate_system

#open_geotiff(path)




