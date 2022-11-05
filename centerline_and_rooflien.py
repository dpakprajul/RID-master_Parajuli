# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 17:24:05 2022

@author: binda
"""
#delete if it doesn't work

from osgeo import gdal, osr
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point, LineString
#from utils import get_wartenberg_boundary, get_static_map_bounds, save_as_geotif
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
from utils import prediction_raster_to_vector, get_progress_string, wkt_to_shape, switch_coordinates
from mask_generation import import_vector_labels
from module_placement import module_placement_options
#from visualization import visualize_module_placement, box_plot_E_gen_TUM_CI    
from azimuth_try import azimuth, area
from azimuth import segment_azimuth
        
    #
ge_img_filepath = "D:\\RID-master\\RID-master\\data\\images_roof_centered_geotiff"
image_path = "D:\\RID-master\\RID-master\\venv\\data\\augment_mask_output_roofline"
save_path =  "D:\\RID-master\\RID-master\\png_to_geotiff\\png"  
 
from utils \
    import get_image_gdf_in_directory, geotif_to_png
    
azimuth_list = []  
    
#for csv files
#file_vector_labels = "D:\\RID-master\\RID-master\\venv\\shapefile\\oilpalm_HarvestedAreaHectares.shp"
file_vector_labels= "D:\\RID-master\\RID-master\\data\\segments_reviewed.csv"
df_labels = gpd.read_file(file_vector_labels)
label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.segment]))
label_geoms = [shapely.ops.transform(switch_coordinates, label) for label in label_geoms]
gdf_labels = gpd.GeoDataFrame(df_labels, geometry=label_geoms)
gdf_labels.crs = 4326
gdf_labels.to_file('C:\\Users\\binda\\Downloads\\check\\6.shp')

g = gdf_labels.iloc[19].geometry
g
a = g.minimum_rotated_rectangle
#simplify = g.simplify(0.5, preserve_topology=True)
#simplify
#s
#s.area
len(a.exterior.coords)
#a
l = g.boundary
l[0]
coords = [c for c in l[0].coords]
coords
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
longest_segment = max(segments, key= lambda x:x.length)
longest_segment



df_labels1 = gpd.read_file(file_vector_labels)
label_geoms1 = list(map(wkt_to_shape, [segment for segment in df_labels1.segment]))
label_geoms1 = [shapely.ops.transform(switch_coordinates, label) for label in label_geoms1]
gdf_labels = gpd.GeoDataFrame(df_labels1, geometry=label_geoms1)
gdf_labels.crs= 4326
g=gdf_labels.iloc[23].geometry
a = g.boundary
a
#a=g.minimum_rotated_rectangle

#segment_azimuth(longest_segment, a)
longest_segment.xy[1][1]
angle = np.arctan((longest_segment.xy[1][1] - longest_segment.xy[1][0]) / (longest_segment.xy[0][1] - longest_segment.xy[0][0])) * 180/np.pi
angle

s = a.centroid
s

def linear_equation(x1, x2, y1, y2, new_x):
    # avoid division through zero
    if (x2-x1) == 0:
        x2 = x2 + 0.000001
    m = (y2-y1) / (x2-x1)
    constant = y1 - m * x1
    new_y = m * new_x + constant

    return new_y


perpendicular = shapely.ops.nearest_points(longest_segment, s)
r=segments[0]
perpendicular[1]
segments1= tuple(r)
#longest_segment.geometry
segments1

#q = list(s.coords)
#q

#gpd.GeoSeries([perpendicular,longest_segment,s]).plot()
#plt.show()
geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
geoseries_shapely
geoseries_shapely.columns = ['geometry']
geoseries_shapely.crs = 4326
#geoseries_shapely.plot()
geoseries_shapely.to_file(r'D:\RID-master\RID-master\venv\data\trash\new\sss.shp')

geoseries_shapely = gpd.GeoSeries(perpendicular, crs=4326)
geoseries_shapely
geoseries_shapely.columns = ['geometry']
geoseries_shapely.crs = 4326
#geoseries_shapely.plot()
geoseries_shapely.to_file(r'D:\RID-master\RID-master\venv\data\trash\new\ppp.shp')

print(perpendicular[1].y)
print(perpendicular[0].y)


dy = perpendicular[1].y - perpendicular[0].y
print(dy)
dx = perpendicular[1].x - perpendicular[0].x
print(dx)

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

azimuth_list
print(len(azimuth_list))
label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat']

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

something = azimuth_to_label_class(azimuth_list[0], label_classes_segments_18)
something


def segment_azimuth(gutters_list, segments_list):
    azimuth_list = []

    for i,g in enumerate(gutters_list):
        print(i)
        print(g)
        print(gutters_list)
        if not g:
            azimuth_list.append(None)
        #elif not isinstance(g, MultiLineString):
            #print('gutters in gutterslist need to be Multilinestring')
        elif len(segments_list)!=len(gutters_list):
            print('number of gutters and segments is unequal')
        else:
            if (g[0].xy[0][1] - g[0].xy[0][0]) == 0: #if delta x is 0, arctan fails, division by zero
                angle = 90
            #else: # calculate arctan
            print(g[0].xy[1][1])
            angle = np.arctan((g[0].xy[1][1] - g[0].xy[1][0]) / (g[0].xy[0][1] - g[0].xy[0][0])) * 180/np.pi
            angle1 = math.degrees(math.atan2((g[0].xy[1][1] - g[0].xy[1][0]) , (g[0].xy[0][1] - g[0].xy[0][0])))
            print(angle)
            print(angle1)
            # get centroid of segment
            s = segments_list[i].centroid
            print(s)

            # define linear equation to calculate potential new extension points for gutter
            def linear_equation(x1, x2, y1, y2, new_x):
                # avoid division through zero
                if (x2-x1) == 0:
                    x2 = x2 + 0.000001
                m = (y2-y1) / (x2-x1)
                constant = y1 - m * x1
                new_y = m * new_x + constant

                return new_y

            # check if centroid is within x-boundaries of gutter
            # if centroid x coordinate is smaller than smallest gutter x coordinate: extend gutter to the left
            if (s.xy[0][0] - g.bounds[0]) < 0:
                new_g_x = g.bounds[0] + 3 * (s.xy[0][0] - g.bounds[0])
                new_g_y = linear_equation(g[0].xy[0][0], g[0].xy[0][1], g[0].xy[1][0], g[0].xy[1][1], new_g_x)
                # create new gutter linestring as extended_gutter, to find nearest point and hence perpendicular
                extended_gutter = LineString(
                    [Point(new_g_x, new_g_y), Point(g[0].xy[0][0], g[0].xy[1][0]), Point(g[0].xy[0][1], g[0].xy[1][1])])
                perpendicular = shapely.ops.nearest_points(extended_gutter, s)
            # if centroid x coordinate is lager than largest gutter x coordinate: extend gutter to the right
            elif (g.bounds[2] - s.xy[0][0]) < 0:
                new_g_x = g.bounds[2] - 3 * (g.bounds[2] - s.xy[0][0])
                new_g_y = linear_equation(g[0].xy[0][0], g[0].xy[0][1], g[0].xy[1][0], g[0].xy[1][1], new_g_x)
                # create new gutter linestring as extended_gutter, to find nearest point and hence perpendicular
                extended_gutter = LineString(
                    [Point(g[0].xy[0][0], g[0].xy[1][0]), Point(g[0].xy[0][1], g[0].xy[1][1]), Point(new_g_x, new_g_y)])
                perpendicular = shapely.ops.nearest_points(extended_gutter, s)
            else:
                perpendicular = shapely.ops.nearest_points(g, s)
                gpd.GeoSeries(perpendicular).plot()
                plt.show()
                print(perpendicular[1].y)
                print(perpendicular[0].y)
    

            # get delta x and y of the perpendicular to adjust azimuth according to the direction of the perpendicular vector
            dy = perpendicular[1].y - perpendicular[0].y
            print(dy)
            dx = perpendicular[1].x - perpendicular[0].x
            print(dx)

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


