# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:28:22 2022

@author: binda
"""

from pyproj import Geod
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
#df_labels= gpd.read_file("D:\\RID-master\\RID-master\\venv\\data\\trash\\oilpalm_HarvestedAreaHectares.shp")
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

#def segment_generation(df_label):
    #gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
    #mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    
segments = []
direction_list = []
gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)


#infile = open('C:\\Users\\binda\\Downloads\\check\\shape_file_original\\mydata.pkl', 'rb')
#mrrs = pickle.load(infile)
print(mrrs)
mrrs = mrrs[0:500]
geod = Geod(ellps="WGS84")
label_classes = gdf_labels.geometry
print(label_classes)

    #filtered = filter(lambda x:x !='None', label_class)
   
   
#label_classes = list(mrrs.geometry) 
segments = []
#for i, mrr in enumerate(mrrs.iloc):
   
min_rot_each = mrrs[99].minimum_rotated_rectangle
boundary = min_rot_each.boundary
coords = [c for c in boundary.coords]
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
segments
geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
#save_directory = "C:\\Users\\binda\\Downloads\\check\\segment"+ "\\" + str(i) + ".shp"
#geoseries_shapely.to_file(save_directory)
save_directory = "C:\\Users\\binda\\Downloads\\check\\segment\\latest_segment.shp"
geoseries_shapely.to_file(save_directory)
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

#g_shapely = gpd.GeoSeries([g[0],g[1],g[2],g[3],mrrs[34],mrrs[35],mrrs[37],mrrs[38],mrrs[39],g0,g1,g2,g3], crs =4326)     
#g_shapely = gpd.GeoSeries([g[0],g[1],g[2],g[3],mrrs[21],mrrs[35],mrrs[37],mrrs[38],mrrs[40],g3], crs =31468)  
#g_shapely.plot() 

centroid_all = []
distance_all_g0 = []
##for first line
#g[0]
for i, segments in enumerate(label_classes):
    #print(label_classes)
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g[0])
    distance_between_points = p1.distance(g[0])
    #p1, p2 = nearest_points(poly, point)
    distance_all_g0.append(distance_between_points)
    #g_pic = gpd.GeoSeries([p1, g[0], centroid_all[i]], crs = 4326)
    #g_pic.plot()
    #print(distance_between_points)
filtered = filter(lambda x: x != 0,distance_all_g0)
#sort = sorted(filtered)
sort = sorted(filtered)
list_distance = sort[0:20]

sort_distance_min = min(list_distance)
sort_distance_min


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
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g[1])
    distance_between_points = p1.distance(g[1])
    #p1, p2 = nearest_points(poly, point)
    distance_all_g1.append(distance_between_points)
    print(distance_between_points)
filtered = filter(lambda x: x != 0,distance_all_g1)
sort = sorted(filtered)
list_distance = sort[0:20]
sort_distance_min1 = min(list_distance)
sort_distance_min1


distance_all_g2 = []
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g[2])
    distance_between_points = p1.distance(g[2])
    #p1, p2 = nearest_points(poly, point)
    distance_all_g2.append(distance_between_points)
    print(distance_between_points)
filtered = filter(lambda x: x != 0,distance_all_g2)
sort = sorted(filtered)
list_distance = sort[0:20]
sort_distance_min2 = min(list_distance)
sort_distance_min2


distance_all_g3 = []
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g[3])
    distance_between_points = p1.distance(g[3])
    #p1, p2 = nearest_points(poly, point)
    distance_all_g3.append(distance_between_points)
    print(distance_between_points)
filtered = filter(lambda x:x !=0, distance_all_g3)
sort = sorted(filtered)
list_distance = sort[0:20]
sort_distance_min3 = min(list_distance)
sort_distance_min3


distances = [sort_distance_min, sort_distance_min1, sort_distance_min2, sort_distance_min3]
max(distances)

distances
max(distances)
if distances[0]==max(distances):
    gutter = g[0]
elif distances[1]==max(distances):
    gutter = g[1]
elif distances[2]==max(distances):
    gutter = g[2]
    print('I am executed')
else:
    gutter = g[3]
    


gutter_shapely = gpd.GeoSeries(gutter, crs=4326)
save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter\\latest_gutter.shp"
gutter_shapely.to_file(save_directory_gutter)


from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

poly = Polygon([(0, 0), (2, 8), (14, 10), (6, 1)])
point = Point(12, 4)
# The points are returned in the same order as the input geometries:
p1, p2 = nearest_points(poly, point)
geoseries_shapely = gpd.GeoSeries([p1,p2,poly, point], crs=4326)
geoseries_shapely.plot()
print(p2.wkt)
