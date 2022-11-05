# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 11:08:19 2022

@author: binda
"""

from shapely.geometry import Point, Polygon
import random
from operator import itemgetter

import geopandas as gpd
import heapq
import shapely
from shapely.geometry import Point, Polygon, MultiPoint
df_labels= gpd.read_file("C:\\Users\\binda\\Downloads\\check\\shape_file_original\\6.shp")
"""
gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
label_classes = list(mrrs.geometry) #it gives the list of all polygon
label_classes
#taking one polygon (polygon of interest)
a = mrrs[300].minimum_rotated_rectangle
b = mrrs[299].minimum_rotated_rectangle

def random_coords(n):
    return [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n)]


polys = [a for i in label_classes[:6]]
polys
point = Point(b.centroid)

min_distance, min_poly = min(((poly.distance(point), poly) for poly in polys), key=itemgetter(0))

min_distance
min_poly
"""
#take polygon from check
i = 19
df_check= gpd.read_file("C:\\Users\\binda\\Downloads\\check\\check\\658.shp")
gdf_check = gpd.GeoDataFrame(df_check, geometry='geometry')
#take 1st element of the dataframe
check_1 = gdf_check.geometry


#take all the segment of the check
mrrs = gdf_check.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
#label_classes = list(mrrs.geometry) #it gives the list of all polygon
#taking one polygon (polygon of interest)
a = mrrs[i].minimum_rotated_rectangle

l = a.boundary
coords = [c for c in l.coords]
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
segments  ##it stores the all the side lines of the polygon ()
geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
save_directory_gutter = "C:\\Users\\binda\\Downloads\\check\\gutter\\latest_gutter1.shp"
geoseries_shapely.to_file(save_directory_gutter)
g = geoseries_shapely
g0 =g[0].centroid        
g1=g[1].centroid      
g2=g[2].centroid       
g3 = g[3].centroid  



df_gutter = gpd.read_file("C:\\Users\\binda\\Downloads\\check\\check_gutter\\658.shp")
gdf_gutter = gpd.GeoDataFrame(df_gutter, geometry='geometry')
mrrs = gdf_gutter.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
label_classes = list(mrrs.geometry) #it gives the list of all polygon

centroid_all = []
distance_all_g0 = []
for i, segments in enumerate(label_classes):
    #print(label_classes)
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g0)
    distance_between_points = p1.distance(g0)
    #p1, p2 = nearest_points(poly, point)
    distance_all_g0.append(distance_between_points)
#filtered = filter(lambda x: x != 0,distance_all_g0)
sort = sorted(distance_all_g0)
list_distance = sort
sort_distance_min = min(list_distance)
sort_distance_min


distance_all_g1 = []
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g1)
    distance_between_points = p1.distance(g1)
    #p1, p2 = nearest_points(poly, point)
    distance_all_g1.append(distance_between_points)
    print(distance_between_points)
#filtered = filter(lambda x: x != 0,distance_all_g1)
sort = sorted(distance_all_g1)
list_distance = sort
sort_distance_min1 = min(list_distance)
sort_distance_min1


distance_all_g2 = []
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g2)
    distance_between_points = p1.distance(g2)
    #p1, p2 = nearest_points(poly, point)
    distance_all_g2.append(distance_between_points)
    print(distance_between_points)
#filtered = filter(lambda x: x != 0,distance_all_g2)
sort = sorted(distance_all_g2)
list_distance = sort
sort_distance_min2 = min(list_distance)
sort_distance_min2


distance_all_g3 = []
for i, segments in enumerate(label_classes):
    centroid_one = label_classes[i]
    centroid_all.append(centroid_one) 
    p1, p2 = shapely.ops.nearest_points(centroid_all[i],g3)
    distance_between_points = p1.distance(g3)
    #p1, p2 = nearest_points(poly, point)
    distance_all_g3.append(distance_between_points)
    print(distance_between_points)
#filtered = filter(lambda x:x !=0, distance_all_g3)
sort = sorted(distance_all_g3)
list_distance = sort
sort_distance_min3 = min(list_distance)
sort_distance_min3


distances = [sort_distance_min, sort_distance_min1, sort_distance_min2, sort_distance_min3]
min(distances)

distances
max(distances)
if distances[0]==min(distances):
    gutter = g[0]
elif distances[1]==min(distances):
    gutter = g[1]
elif distances[2]==min(distances):
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
