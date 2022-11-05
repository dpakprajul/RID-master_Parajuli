# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:57:21 2022

@author: binda
"""


import geopandas as gpd
import shapely
import math
from shapely.ops import nearest_points
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, Point, box, MultiLineString, LineString
import matplotlib.pyplot as plt
from utils import convert_between_latlon_and_pixel, wkt_to_shape, azimuth_to_label_class, \
    switch_coordinates, get_progress_string
from definitions import \
    LABEL_CLASSES_SUPERSTRUCTURES, \
    LABEL_CLASSES_6


#azimuth_list = []
#file_vector_labels = "D:\\RID-master\\RID-master\\data\\countries.csv"
##df_labels = pd.read_csv(file_vector_labels)
#label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.geometry]))

#df_labels1 = pd.read_csv(file_vector_labels)
#df_labels = df_labels[df_labels.geometry.notna()]
#label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.geometry]))
#mrr = label_geoms[1].minimum_rotated_rectangle
#mrr
#p= label_geoms[1]
#p
#a = p.minimum_rotated_rectangle
#a



def _azimuth(point1, point2):
    """azimuth between 2 points (interval 0 - 180)"""
    import numpy as np

    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

def _dist(a, b):
    """distance between points"""
    import math

    return math.hypot(b[0] - a[0], b[1] - a[1])

def azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    print(axis1)
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return float("{:.2f}".format(az))

def area(mrr):

    return mrr.geometry.area
#azimuth(a)

###for csv files
#file_vector_labels = "D:\\RID-master\\RID-master\\venv\\shapefile\\oilpalm_HarvestedAreaHectares.shp"
#file_vector_labels= "D:\\RID-master\\RID-master\\data\\segments_reviewed.csv"
#df_labels = gpd.read_file(file_vector_labels)
#label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.segment]))
#label_geoms = [shapely.ops.transform(switch_coordinates, label) for label in label_geoms]
#gdf_labels = gpd.GeoDataFrame(df_labels, geometry=label_geoms)
#gdf_labels.crs = 4326



#mrr = gdf_labels.geometry.iloc[1].minimum_rotated_rectangle
#print(azimuth(mrr))

#mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
#mrrs
#df_labels['az'] = mrrs.apply(azimuth)

#ax = df_labels.plot('az', aspect =1, legend=True)


#b= mrrs.boundary.plot(ax=ax, alpha=0.5)


import matplotlib.colors as colors
#for shp files
file_vector_labels= "D:\\RID-master\\RID-master\\venv\\shapefile\\oilpalm_HarvestedAreaHectares.shp"
df_labels = gpd.read_file(file_vector_labels)
#label_geoms = list(map(wkt_to_shape, [segment for segment in df_labels.segment]))
#label_geoms = [shapely.ops.transform(switch_coordinates, label) for label in label_geoms]
#gdf_labels = gpd.GeoDataFrame(df_labels, geometry=label_geoms)
#gdf_labels.crs = 4326
gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
df_labels.crs=4326


color_dict = {'Africa':'orange', 'Antarctica':'pink', 'East':'white', 
              'Europe':'green', 'North America':'brown',
              'Oceania':'blue', 'Seven seas (open ocean)':'gray',
              'North':'red'}

#mrr = df_labels.geometry.iloc[1].minimum_rotated_rectangle
#print(azimuth(mrr))

mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
mrrs
df_labels['az'] = mrrs.apply(azimuth)
df_labels
ax = df_labels.plot('az', aspect =1, legend=True)
b= mrrs.boundary.plot(ax=ax, alpha=0.5)






#l = mrr.boundary
#l
#a[0].xy[0][1]
#longest_segment = max(a, key= lambda x:x.length)
#longest_segment
#p1,p2 = [c for c in longest_segment.coords]
#p1
#p2
#angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
#angle

#coords = [c for c in l.coords]
#segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
#segments
#m = max(segments, key= lambda x:x.length)

#p1,p2 = [c for c in m.coords]
#p1
#p2
#angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
#angle


#s = l.centroid
#s

#perpendicular = shapely.ops.nearest_points(l, s)
#perpendicular
#gpd.GeoSeries(perpendicular).plot()
#plt.show()
#print(perpendicular[1].y)
#print(perpendicular[0].y)


#dy = perpendicular[1].y - perpendicular[0].y
#print(dy)
#dx = perpendicular[1].x - perpendicular[0].x
#print(dx)

#if angle >= 0 and dx > 0:
#    angle = 180-angle
#elif angle >= 0 and dx <= 0:
#    angle = angle * -1
#elif angle < 0 and dx <= 0:
#    angle = (180+angle)*-1
#elif angle < 0 and dx > 0:
#    angle = angle * -1
#else:
#    print('unexpected values in azimuth angle detection')
#azimuth_list.append(angle)
#azimuth_list

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
    

###
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


#segment_azimuth(m, l)












