# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 01:52:40 2022

@author: binda
"""

from utils \
    import get_image_gdf_in_directory, geotif_to_png
#from azimuth_try import azimuth, area
from module_placement import module_placement_options
from mask_generation import import_vector_labels
from utils import prediction_raster_to_vector, get_progress_string, prediction_raster_to_vector1
from definitions import LABEL_CLASSES_SUPERSTRUCTURES, LABEL_CLASSES_SEGMENTS, IMAGE_SHAPE, \
    FILE_VECTOR_LABELS_SEGMENTS, FILE_VECTOR_LABELS_SUPERSTRUCTURES, DIR_IMAGES_PNG, \
    df_technical_potential_LUT, EPSG_METRIC, DIR_PREDICTIONS, LABEL_CLASSES_6, LABEL_CLASSES_PV_AREAS, LABEL_CLASSES_ROOFLINE, OUTPUT_FROM_PREDICTION, \
    DIR_IMAGES_GEOTIFF_TRASH, COMPARISON_GUTTER
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
    PNG_TO_GEOTIFF, PNG_TO_GEOTIFF1, DIR_CREATE_AND_DELETE, DIR_PREDICTIONS1,COMPARISON_ORIG
#from orientation_evaluation import orient_evaluation, segment_generation
from model_evaluation import visualize_prediction_confusion_matrix
from definitions import \
    LABEL_CLASSES_SUPERSTRUCTURES,\
    LABEL_CLASSES_SUPERSTRUCTURES1
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
from sklearn.metrics import confusion_matrix, accuracy_score
shapely.speedups.disable()
#from gold_copy import segment_generation, segment_generation1
#from visualization import visualize_module_placement, box_plot_E_gen_TUM_CI

import cv2
import numpy as np
from PIL import Image

def azimuth_to_label_class(az, label_classes):
     label_classes = label_classes[:-1]
     if np.isnan(az):
         az_class = "flat"
     else:
         surplus_angle = 360 / len(label_classes) / 2
         az = az + 180 + surplus_angle
         if az > 360:
             az -= 360
         az_id = int(np.ceil(az / (360 / len(label_classes))) - 1)
         az_class = label_classes[az_id]
     return az_class
 
def segment_azimuth(gutters_list, segments_list):
    azimuth_list = []

    for i,g in enumerate(gutters_list):
        if not g:
            azimuth_list.append(None)
        else:
            if (g.xy[0][1] - g.xy[0][0]) == 0: #if delta x is 0, arctan fails, division by zero
                angle = 90
                azimuth_list.append(angle)
                return azimuth_list 
            else: # calculate arctan
                angle = np.arctan((g.xy[1][1] - g.xy[1][0]) / (g.xy[0][1] - g.xy[0][0])) * 180/np.pi
                angle1 = math.degrees(math.atan2((g.xy[1][1] - g.xy[1][0]) , (g.xy[0][1] - g.xy[0][0])))
                # get centroid of segment
                s = segments_list[i].centroid
                perpendicular = shapely.ops.nearest_points(g, s)

                # get delta x and y of the perpendicular to adjust azimuth according to the direction of the perpendicular vector
                dy = perpendicular[1].y - perpendicular[0].y
                dx = perpendicular[1].x - perpendicular[0].x
            

                if angle >= 0 and dx > 0:
                    angle = 180-angle
                elif angle >= 0 and dx <= 0:
                    angle = angle * -1
                elif angle < 0 and dx <= 0:
                    angle = (180+angle)*-1
                elif angle < 0 and dx > 0:
                    angle = angle * -1
                else:
                    print('unexpected values in azimuth angle detection')
                azimuth_list.append(angle)
                return azimuth_list 

def segment_generation(df_label1, mask_id1):
    
    df_label = gpd.GeoDataFrame(df_label1, geometry='geometry')
    mrrs = df_label.geometry
    segments = []
    direction_list = []
   

   
    label_classes = list(mrrs.geometry) 
   #label_classes = list(mrrsp.geometry)
    segments = []
    azimuth_list = []
   #geoser = []
   
   
   
    for i, mrr in enumerate(mrrs.iloc):
       min_rot_each = mrrs[i].minimum_rotated_rectangle
       boundary = min_rot_each.boundary
       try:
           coords = [c for c in boundary.coords]
   # block raising an exception
       except:
           pass # doing nothing on exception
       segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
       geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
       
       #geoser.append(segments)
       #save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(i) + ".shp"
       #geoseries_shapely.to_file(save_directory)
       #save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(i) + ".shp"
       #geoseries_shapely.to_file(save_directory)
       segment_lists = []
       for seg, segment_list in enumerate(segments):
           segment_list = segments[seg].length
           segment_lists.append(segment_list)
       geoseries_shapely['length'] = segment_lists
       g = geoseries_shapely
       g0 =g[0].centroid        
       g1=g[1].centroid      
       g2=g[2].centroid       
       g3 = g[3].centroid  
       #g_shapely = gpd.GeoSeries([g[0],g[1],g[2],g[3],mrrs[21],mrrs[35],mrrs[37],mrrs[38],mrrs[40],g3], crs =31468)  
       #g_shapely.plot() 

       centroid_all = []
       distance_all_g0 = []
       ##for first line
       #g[0]
       for id1, segments in enumerate(label_classes):
           centroid_one = label_classes[id1]
           centroid_all.append(centroid_one) 
           p1, p2 = shapely.ops.nearest_points(centroid_all[id1],g0)
           distance_between_points = p1.distance(g0)
           #p1, p2 = nearest_points(poly, point)
           distance_all_g0.append(distance_between_points)
           
       filtered = filter(lambda x: x != 0,distance_all_g0)
       sort = sorted(filtered)
       list_distance = sort[0:20]
       sort_distance_min = min(list_distance)
       #sort_distance_min
       #distance_all_g1 = []
       #for i, segments in enumerate(label_classes):
           #centroid_one = label_classes[i].centroid
           #centroid_all.append(centroid_one)
           #distance_between_points = g1.distance(centroid_all[i])
           #distance_all_g1.append(distance_between_points)

       #sort1 = sorted(distance_all_g1)
       #list_distance1 = sort1[0:20]

       #sort_distance_min1 = min(list_distance1)
       distance_all_g1 = []
       for id2, segments in enumerate(label_classes):
           centroid_one = label_classes[id2]
           centroid_all.append(centroid_one) 
           p1, p2 = shapely.ops.nearest_points(centroid_all[id2],g1)
         
           distance_between_points = p1.distance(g1)
           #p1, p2 = nearest_points(poly, point)
           distance_all_g1.append(distance_between_points)
           #print(distance_between_points)
       filtered = filter(lambda x: x != 0,distance_all_g1)
       sort = sorted(filtered)
       list_distance = sort[0:20]
       sort_distance_min1 = min(list_distance)

       distance_all_g2 = []
       for id3, segments in enumerate(label_classes):
           centroid_one = label_classes[id3]
           centroid_all.append(centroid_one) 
           p1, p2 = shapely.ops.nearest_points(centroid_all[id3],g2)
           
           distance_between_points = p1.distance(g2)
           #p1, p2 = nearest_points(poly, point)
           distance_all_g2.append(distance_between_points)
           #print(distance_between_points)
       filtered = filter(lambda x: x != 0,distance_all_g2)
       sort = sorted(filtered)
       list_distance = sort[0:20]
       sort_distance_min2 = min(list_distance)



       distance_all_g3 = []
       for id4, segments in enumerate(label_classes):
           centroid_one = label_classes[id4]
           centroid_all.append(centroid_one) 
           p1, p2 = shapely.ops.nearest_points(centroid_all[id4],g3)
          
           distance_between_points = p1.distance(g3)
           #p1, p2 = nearest_points(poly, point)
           distance_all_g3.append(distance_between_points)
           #print(distance_between_points)
       filtered = filter(lambda x: x != 0,distance_all_g3)
       sort = sorted(filtered)
       list_distance = sort[0:20]
       sort_distance_min3 = min(list_distance)

       distances = [sort_distance_min, sort_distance_min1, sort_distance_min2, sort_distance_min3]
     
       if distances[0]==max(distances):
           gutter = g[0]
       elif distances[1]==max(distances):
           gutter = g[1]
       elif distances[2]==max(distances):
           gutter = g[2]
       else:
           gutter = g[3]
           


       gutter_shapely = gpd.GeoSeries(gutter, crs=4326)
       #save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter"+ "\\" + str(i) + ".shp"
       #gutter_shapely.to_file(save_directory_gutter)
       
       azimuth = segment_azimuth(gutter_shapely, g)
       az = azimuth[0]
       label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat']

      
       direction = azimuth_to_label_class(az, label_classes_segments_18)
       azimuth_list.append(az)
       direction_list.append(direction)
       
       #for i, m in enumerate(geoser):
           ##save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(i) + ".shp"
           #m[i].to_file(save_directory)
        
    return direction_list, azimuth_list

def segment_generation1(df_label1, df_label2, mask_id1):
    
    df_label1 = gpd.GeoDataFrame(df_label1, geometry='geometry')
    #df_label1 = df_label1.explode()
    mrrs = df_label1.geometry
    segments = []
    direction_list = []
    azimuth_list = []
    #geoser = []
    
    df_label2 = gpd.GeoDataFrame(df_label2, geometry='geometry')
    mrrs2 = df_label2.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    label_classes = list(mrrs2.geometry) #it gives the list of all polygon
    
    
    for i, mrr in enumerate(mrrs.iloc): 
        min_rot_each = mrrs[i].minimum_rotated_rectangle
        min_rot_each1 = gpd.GeoSeries(min_rot_each, crs= 4326)
        #save_directory = "C:\\Users\\binda\\Downloads\\check\\mrr"+ "\\" + str(mask_id1)+ "_" + str(i) + ".shp"
        #min_rot_each1.to_file(save_directory)
        
        boundary = min_rot_each.boundary
        try:
            coords = [c for c in boundary.coords]
    # block raising an exception
        except:
            pass # doing nothing on exception
        
        segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
        geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
        
        #save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(mask_id1)+ "_" + str(i) + ".shp"
       # geoseries_shapely.to_file(save_directory)
        
        segment_lists = []
        for seg, segment_list in enumerate(segments):
            segment_list = segments[seg].length
            segment_lists.append(segment_list)
        segment_lists
        geoseries_shapely['length'] = segment_lists
        g = geoseries_shapely
        g0 =g[0].centroid        
        g1=g[1].centroid      
        g2=g[2].centroid       
        g3 = g[3].centroid
        
      

        centroid_all = []
        distance_all_g0 = []
        ##for first line
        #g[0]
        for id1, segments in enumerate(label_classes):
            #print(label_classes)
            centroid_one = label_classes[id1]
            centroid_all.append(centroid_one) 
            p1, p2 = shapely.ops.nearest_points(centroid_all[id1],g0)
            distance_between_points = p1.distance(g[0])
            #p1, p2 = nearest_points(poly, point)
            distance_all_g0.append(distance_between_points)
        #filtered = filter(lambda x: x != 0,distance_all_g0)
        sort = sorted(distance_all_g0)
        list_distance = sort
        sort_distance_min = min(list_distance)
        
       
        distance_all_g1 = []
        for id2, segments in enumerate(label_classes):
            centroid_one = label_classes[id2]
            centroid_all.append(centroid_one) 
            p1, p2 = shapely.ops.nearest_points(centroid_all[id2],g1)
            distance_between_points = p1.distance(g[1])
            #p1, p2 = nearest_points(poly, point)
            distance_all_g1.append(distance_between_points)
            
        #filtered = filter(lambda x: x != 0,distance_all_g1)
        sort = sorted(distance_all_g1)
        list_distance = sort
        sort_distance_min1 = min(list_distance)

        distance_all_g2 = []
        for id3, segments in enumerate(label_classes):
            centroid_one = label_classes[id3]
            centroid_all.append(centroid_one) 
            p1, p2 = shapely.ops.nearest_points(centroid_all[id3],g2)
            distance_between_points = p1.distance(g[2])
            #p1, p2 = nearest_points(poly, point)
            distance_all_g2.append(distance_between_points)
            
        #filtered = filter(lambda x: x != 0,distance_all_g2)
        sort = sorted(distance_all_g2)
        list_distance = sort
        sort_distance_min2 = min(list_distance)



        distance_all_g3 = []
        for id4, segments in enumerate(label_classes):
            centroid_one = label_classes[id4]
            centroid_all.append(centroid_one) 
            p1, p2 = shapely.ops.nearest_points(centroid_all[id4],g3)
            distance_between_points = p1.distance(g[3])
            #p1, p2 = nearest_points(poly, point)
            distance_all_g3.append(distance_between_points)
            
        #filtered = filter(lambda x:x !=0, distance_all_g3)
        sort = sorted(distance_all_g3)
        list_distance = sort
        sort_distance_min3 = min(list_distance)
       

        distances = [sort_distance_min, sort_distance_min1, sort_distance_min2, sort_distance_min3]
      
        if distances[0]==min(distances):
            gutter = g[0]
        elif distances[1]==min(distances):
            gutter = g[1]
        elif distances[2]==min(distances):
            gutter = g[2]
        else:
            gutter = g[3]
            


        g=gutter
        gutter_shapely = gpd.GeoSeries(g, crs=4326)
        #save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter"+ "\\" + str(mask_id1)+ "_" + str(i) + ".shp"
        #gutter_shapely.to_file(save_directory_gutter)
        
        df1 =gutter_shapely.geometry[0].coords[0]
        df2= gutter_shapely.geometry[0].coords[1]

        #create a linestring from 2 tuples
        #df1 = (11.994593070128408, 48.400156441116394)
        #df2 =(11.994587743377071, 48.400215812129)

        #make a linestring from the above tuple
        line = LineString([df1, df2])

        parallel = line.parallel_offset(0.005, 'right', resolution=16, join_style=1, mitre_limit=5.0)
        perp = LineString([line.centroid, parallel.centroid])

        
        #take the first point of the first line
        azimuth = math.degrees(math.atan2((perp.xy[0][1] - perp.xy[0][0]),(perp.xy[1][1] - perp.xy[1][0])))
        angle = azimuth
        #convert the azimuth to positive
        azimuth = azimuth + 360 if azimuth < 0 else azimuth
        
    
        

        #find the orientation in terms of direction like 'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat'
        if azimuth >= 348.75 or azimuth < 11.25:
            direction = 'N'
        elif azimuth >= 11.25 and azimuth < 33.75:
            direction = 'NNE'
        elif azimuth >= 33.75 and azimuth < 56.25:
            direction = 'NE'
        elif azimuth >= 56.25 and azimuth < 78.75:
            direction = 'ENE'
        elif azimuth >= 78.75 and azimuth < 101.25:
            direction = 'E'
        elif azimuth >= 101.25 and azimuth < 123.75:
            direction = 'ESE'
        elif azimuth >= 123.75 and azimuth < 146.25:
            direction = 'SE'
        elif azimuth >= 146.25 and azimuth < 168.75:
            direction = 'SSE'
        elif azimuth >= 168.75 and azimuth < 191.25:
            direction = 'S'
        elif azimuth >= 191.25 and azimuth < 213.75:
            direction = 'SSW'
        elif azimuth >= 213.75 and azimuth < 236.25:
            direction = 'SW'
        elif azimuth >= 236.25 and azimuth < 258.75:
            direction = 'WSW'
        elif azimuth >= 258.75 and azimuth < 281.25:
            direction = 'W'
        elif azimuth >= 281.25 and azimuth < 303.75:
            direction = 'WNW'
        elif azimuth >= 303.75 and azimuth < 326.25:
            direction = 'NW'
        elif azimuth >= 326.25 and azimuth < 348.75:
            direction = 'NNW'
        else :
            direction = 'flat'
        

       
        azimuth_list.append(azimuth)

       
        
        direction_list.append(direction)
        
    return direction_list, azimuth_list

list1_append = []
list2_append = []
az1_append = []
az2_append = []
def orient_evaluation(df_labels1, df_labels2):
    

    df_labels1
    df_labels2
    gdf_check1 = gpd.GeoDataFrame(df_labels1, geometry='geometry')
    gdf_check2 = gpd.GeoDataFrame(df_labels2, geometry='geometry')

    #abc=gpd.sjoin(df_labels1, df_labels2, how = 'left')
    abc = gpd.overlay(df_labels1, df_labels2, how='intersection', keep_geom_type=False, make_valid=True)
    gdf_abc= gpd.GeoDataFrame(abc, geometry='geometry')
    abcd = abc.fillna('w')

   
    for i, segment in enumerate(abcd.iloc):
       # list1 = segment[3]
       # list2 = segment[7]
       #change the index of segment accordingly
        list1 = segment[2]
        list2 = segment[6]
        
        az1 = segment[3]
        az2 = segment[7]
        
        list1_append.append(list1)
        list2_append.append(list2)
        
        az1_append.append(az1)
        az2_append.append(az2)
        
        a = confusion_matrix(list1_append, list2_append)
        #visualize_prediction_confusion_matrix(a, LABEL_CLASSES_SEGMENTS.values()) 

    return list1_append, list2_append, az1_append, az2_append
def orientation():
    prediction_mask_filename = os.listdir(PNG_TO_GEOTIFF)
    prediction_mask_filepath = [os.path.join(
        PNG_TO_GEOTIFF, file) for file in prediction_mask_filename]
    prediction_mask = [cv2.imread(prediction, 0)
                       for prediction in prediction_mask_filepath]
    
    prediction_mask_filename1 = os.listdir(PNG_TO_GEOTIFF1)
    prediction_mask_filepath1 = [os.path.join(
        PNG_TO_GEOTIFF1, file) for file in prediction_mask_filename1]
    prediction_mask1 = [cv2.imread(prediction, 0)
                       for prediction in prediction_mask_filepath1]
    
    prediction_mask_filename3 = os.listdir(COMPARISON_ORIG)
    prediction_mask_filepath3 = [os.path.join(COMPARISON_ORIG, file) for file in prediction_mask_filename3]
    prediction_mask3 = [cv2.imread(prediction, 0) for prediction in prediction_mask_filepath3]
    
    prediction_mask_filename4 = os.listdir(COMPARISON_GUTTER)
    prediction_mask_filepath4 = [os.path.join(
        COMPARISON_GUTTER, file) for file in prediction_mask_filename4]
    prediction_mask4 = [cv2.imread(prediction, 0)
                       for prediction in prediction_mask_filepath4]


    with open("data\\gdf_image_boundaries.pkl", 'rb') as f:
        gdf_images = pickle.load(f)
    gdf_images.id = gdf_images.id.astype(int)

    gdf_predictions = []
    dir_lists = []
    dir_lists1 = []
    
    

    # # c) convert raster data of superstructures to vector data
    for i, (mask, mask1, mask3, mask4) in enumerate(zip(prediction_mask, prediction_mask1, prediction_mask3, prediction_mask4)):
        mask_id = prediction_mask_filename[i][:-4]
        gdf_image = gdf_images[gdf_images.id == int(mask_id)]
        image_bbox = gdf_image.geometry.iloc[0]
        gdf_predictions = prediction_raster_to_vector(
            mask, mask_id, image_bbox, LABEL_CLASSES_PV_AREAS, IMAGE_SHAPE)
        gdf_predictions = gdf_predictions.iloc[2: , :]
        gdf_predictions.reset_index(drop=True, inplace=True)
        gdf_labels = gpd.GeoDataFrame(gdf_predictions, geometry='geometry')
        gdf_labels.crs = 4326
        
        
        mask_id1 = prediction_mask_filename1[i][:-4]
        gdf_image1 = gdf_images[gdf_images.id == int(mask_id1)]
        image_bbox1 = gdf_image1.geometry.iloc[0]
        gdf_predictions1 = prediction_raster_to_vector(
            mask1, mask_id1, image_bbox1, LABEL_CLASSES_PV_AREAS, IMAGE_SHAPE)
        gdf_predictions1 = gdf_predictions1[:-1]
        gdf_predictions1.reset_index(drop=True, inplace=True)
        
        gdf_labels1 = gpd.GeoDataFrame(gdf_predictions1, geometry='geometry')
        gdf_labels1.crs = 4326
        #min_rot_each = gdf_labels.minimum_rotated_rectangle
        #save_directory1 = "C:\\Users\\binda\\Downloads\\check\\check_gutter"+ "\\" + str(mask_id1) + ".shp"
        #gdf_labels1.to_file(save_directory1)
        
        mask_id3 = prediction_mask_filename3[i][:-4]
        gdf_image3 = gdf_images[gdf_images.id == int(mask_id3)]
        image_bbox3 = gdf_image3.geometry.iloc[0]
        gdf_predictions3 = prediction_raster_to_vector1(
            mask3, mask_id3, image_bbox3, LABEL_CLASSES_ROOFLINE, IMAGE_SHAPE)
        gdf_predictions3 = gdf_predictions3.iloc[2: , :]
        gdf_predictions3.reset_index(drop=True, inplace=True)
        gdf_labels3 = gpd.GeoDataFrame(gdf_predictions3, geometry='geometry')
        gdf_labels3.crs = 4326
       
        
        mask_id4 = prediction_mask_filename4[i][:-4]
        gdf_image4 = gdf_images[gdf_images.id == int(mask_id4)]
        image_bbox4 = gdf_image4.geometry.iloc[0]
        gdf_predictions4 = prediction_raster_to_vector(
            mask4, mask_id4, image_bbox4, LABEL_CLASSES_ROOFLINE, IMAGE_SHAPE)
        gdf_predictions4 = gdf_predictions4[:-1]
        gdf_predictions4.reset_index(drop=True, inplace=True)
        # gdf_predictions4=gdf_predictions4.explode()
        gdf_labels4 = gpd.GeoDataFrame(gdf_predictions4, geometry='geometry')
        gdf_labels4.crs = 4326
        
        
        dir_list, az_list = segment_generation1(gdf_labels, gdf_labels1, mask_id1)
        dir_list1, az_list1 = segment_generation(gdf_labels3, mask_id4)
        
        dir_lists.append(dir_list)
        dir_lists1.append(dir_list1)
        gdf_labels['direction']= dir_list
        gdf_labels3['direction'] = dir_list1
        gdf_labels['azimuth']= az_list
        gdf_labels3['azimuth'] = az_list1
        """
        
        try:
            save_directory = "C:\\Users\\binda\\Downloads\\check\\check"+ "\\" + str(mask_id) + ".shp"
            gdf_labels.to_file(save_directory)
        except:
            pass
        try:
            save_directory = "C:\\Users\\binda\\Downloads\\check\\comparison_segment"+ "\\" + str(mask_id) + ".shp"
            gdf_labels3.to_file(save_directory)
        except:
            pass
            
          """  
        
        list1_append, list2_append, az1_append, az2_append = orient_evaluation(gdf_labels, gdf_labels3)
        #a = confusion_matrix(list1_append, list2_append)
        #visualize_prediction_confusion_matrix(a, LABEL_CLASSES_SEGMENTS.values())
    a = confusion_matrix(list1_append, list2_append, normalize="true")
    visualize_prediction_confusion_matrix(a, LABEL_CLASSES_SEGMENTS.values())
    print(accuracy_score(list1_append, list2_append))
    print(confusion_matrix(list1_append, list2_append, normalize="true").diagonal())
    
    
    with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list1.txt", "w") as output:
        output.write(str(list1_append))
    with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list2.txt", "w") as output:
        output.write(str(list2_append))
    with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\az_list1.txt", "w") as output:
        output.write(str(az1_append))
    with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\az_list2.txt", "w") as output:
        output.write(str(az2_append))
           
        
    
    #dir_lists
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



