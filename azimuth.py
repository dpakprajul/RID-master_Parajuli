# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 20:23:32 2022

@author: binda
"""
from pv_potential import pv_potential_analysis
import geopandas as gpd
import shapely
import math
from shapely.ops import nearest_points, cascaded_union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, Point, box, MultiLineString, LineString

from utils import convert_between_latlon_and_pixel, wkt_to_shape, azimuth_to_label_class, \
    switch_coordinates, get_progress_string
from definitions import \
    LABEL_CLASSES_SUPERSTRUCTURES, \
    LABEL_CLASSES_6
#pv_potential_analysis()
df= gpd.read_file("D:\\RID-master\\RID-master\\venv\\shapefile\\oilpalm_HarvestedAreaHectares.shp")
g = df.iloc[0].geometry
g
a = g.minimum_rotated_rectangle
l = a.boundary
l
coords = [c for c in l.coords]
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
longest_segment = max(segments, key= lambda x:x.length)
longest_segment
p1,p2 = [c for c in longest_segment.coords]
p1
p2
angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
angle
label_classes=LABEL_CLASSES_6
label_classes

df['angle'] = angle

#df.insert(3, 'angle', angle)
df
label_classes = list(label_classes.values())
label_classes
#class_data = [azimuth_to_label_class(az, label_classes) for az in df.angle]
#class_data
file_vector_labels = "D:\\RID-master\\RID-master\\data\\pv_areas_reviewed.csv"
df_labels = pd.read_csv(file_vector_labels)
#df_labels1 = pd.read_csv(file_vector_labels)
df_labels = df_labels[df_labels.gutters.notna()]

class_data = [label_classes for label_classes in df_labels.roof_type]
class_data
df_labels.insert(df_labels.shape[1], 'class_type', class_data, True)
df_labels
#gutters_list= df_labels["gutters"]
#label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.gutters]))
#label_geoms
#gutters_list
label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.gutters]))
label_geoms
gpd.GeoSeries(label_geoms).plot()
plt.show()
print(df_labels.area) 
print(df_labels.gutters)   
#segments_list = df_labels["area"]
#segments_list

df_labels = df_labels[df_labels.area.notna()]
print(df_labels)
class_data1 = [label_classes for label_classes in df_labels.roof_type]    
class_data1
df_labels.insert(df_labels.shape[1], 'class_type', class_data1, True)

label_geoms1 = list(map(wkt_to_shape, [segment for segment in df_labels.area]))
print(len(label_geoms))
print(len(label_geoms))
gpd.GeoSeries(label_geoms1).plot()
plt.show()

new_shape = shapely.ops.cascaded_union([label_geoms[1], label_geoms1[0]])


gpd.GeoSeries(new_shape).plot()
plt.show()

from shapely.ops import nearest_points
def segment_azimuth(gutters_list, segments_list):
    azimuth_list = []

    for i,g in enumerate(gutters_list):
        print(i)
        print(g)
        print(gutters_list)
        if not g:
            azimuth_list.append(None)
        elif not isinstance(g, MultiLineString):
            print('gutters in gutterslist need to be Multilinestring')
        #elif len(segments_list)!=len(gutters_list):
            #print('number of gutters and segments is unequal')
        else:
            if (g[0].xy[0][1] - g[0].xy[0][0]) == 0: #if delta x is 0, arctan fails, division by zero
                angle = 90
            else: # calculate arctan
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


segment_azimuth(label_geoms, label_geoms1)

