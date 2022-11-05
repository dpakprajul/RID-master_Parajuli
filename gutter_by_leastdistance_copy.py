# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 10:42:13 2022

@author: binda
"""

import geopandas as gpd
import heapq
import shapely
import pickle
from shapely.geometry import Point, Polygon, MultiPoint, LineString
import math
import matplotlib.pyplot as plt
#from centerline_and_rooflien import segment_azimuth
import numpy as np
df_labels= gpd.read_file("C:\\Users\\binda\\Downloads\\check\\shape_file_original\\6.shp")
#save_directory = "C:\\Users\\binda\\Downloads\\check"+ "\\" + str(i) + ".shp"
#save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter"+ "\\" + str(i) + ".shp"
azimuth_list = []

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
            else: # calculate arctan
                angle = np.arctan((g.xy[1][1] - g.xy[1][0]) / (g.xy[0][1] - g.xy[0][0])) * 180/np.pi
                angle1 = math.degrees(math.atan2((g.xy[1][1] - g.xy[1][0]) , (g.xy[0][1] - g.xy[0][0])))
                # get centroid of segment
                s = segments_list[i].centroid
                perpendicular = shapely.ops.nearest_points(g, s)
                #gpd.GeoSeries(perpendicular).plot()
                #plt.show()
            
    

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
                
   

def segment_generation(df_label):
    #gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
    #mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    segments = []
    direction_list = []
    
    infile = open('C:\\Users\\binda\\Downloads\\check\\shape_file_original\\mydata.pkl', 'rb')
    mrrs = pickle.load(infile)
    #print(mrrs)
    mrrs = mrrs[0:1000]

    
    label_classes = list(mrrs.geometry) 
    segments = []
    for i, mrr in enumerate(mrrs.iloc):
       
        min_rot_each = mrrs[i].minimum_rotated_rectangle
        boundary = min_rot_each.boundary
        coords = [c for c in boundary.coords]
        segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
        geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
        save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(i) + ".shp"
        geoseries_shapely.to_file(save_directory)
        segment_lists = []
        #segment = segment_generation(df_labels)
        for seg, segment_list in enumerate(segments):
            segment_list = segments[seg].length
            segment_lists.append(segment_list)
        geoseries_shapely['length'] = segment_lists
        g = geoseries_shapely
        
        lineSegments= []
        if (g[0].length>g[1].length and g[0].length>g[3].length):
            lineSegment = geoseries_shapely[0]
            lineSegments.append(lineSegment)
            lineSegments.append(geoseries_shapely[2])
            #print('I am hero')
        elif(g[1].length>g[2].length):
            lineSegment = geoseries_shapely[1]
            lineSegments.append(lineSegment)
            lineSegments.append(geoseries_shapely[3])
            
        elif(g[2].length>g[3].length):
            lineSegment = geoseries_shapely[2]
            lineSegments.append(lineSegment)
            lineSegments.append(geoseries_shapely[0])
        elif(g[3].length>g[1].length):
            lineSegment = geoseries_shapely[3]
            lineSegments.append(lineSegment)
            lineSegments.append(geoseries_shapely[1])
        
        centroid_all = []
        distance_all = []
        
        for id1, class_id in enumerate(label_classes):
            centroid_one = label_classes[id1].centroid
            centroid_all.append(centroid_one)
            distance_between_points = lineSegments[0].distance(centroid_all[id1])
            distance_all.append(distance_between_points)
            
        sort_distance_min = sorted(distance_all)  #sorting distance in ascending order
        list_distance = sort_distance_min[0:1]   #taking only 4 elements out of distance_all
        max_of_min = max(list_distance) 
        
        centroid_all1 = []
        distance_all1 = []

        for id2, class_id in enumerate(label_classes):
            centroid_one1 = label_classes[id2].centroid
            centroid_all1.append(centroid_one1)
            distance_between_points1 = lineSegments[1].distance(centroid_all1[id2])
            distance_all1.append(distance_between_points1)
        sort_distance_min1 = sorted(distance_all1)
        list_distance1 = sort_distance_min1[0:1]
        max_of_min1 = max(list_distance1)
        
        if max_of_min < max_of_min1:
            gutter = lineSegments[1]
        else:
            gutter = lineSegments[0]
        
        gutter_shapely = gpd.GeoSeries(gutter, crs=4326)
        save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter"+ "\\" + str(i) + ".shp"
        gutter_shapely.to_file(save_directory_gutter)
        
        
    
        azimuth = segment_azimuth(gutter_shapely, g)
        az = azimuth[0]
        label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat']

       
        direction = azimuth_to_label_class(az, label_classes_segments_18)
        direction_list.append(direction)
        #print(direction_list)
    return direction_list

    
        #save_directory = "C:\\Users\\binda\\Downloads\\check"+ "\\" + str(i) + ".shp"
        #geoseries_shapely.to_file(save_directory)
    
                 
dir_list = segment_generation(df_labels)
print(dir_list)

with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list1.txt", "w") as output:
    output.write(str(dir_list))

dir_list
        
from collections import Counter

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

print(Most_Common(dir_list))



import collections
# intializing the arr
arr = dir_list
# getting the elements frequencies using Counter class
elements_count = collections.Counter(arr)
# printing the element and the frequency
for key, value in elements_count.items():
   print(f"{key}: {value}")
