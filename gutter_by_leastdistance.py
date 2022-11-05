# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:40:22 2022

@author: binda
"""

import geopandas as gpd
import heapq
import shapely
from shapely.geometry import Point, Polygon, MultiPoint
df_labels= gpd.read_file("C:\\Users\\binda\\Downloads\\check\\shape_file_original\\6.shp")

gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')
mrrs = gdf_labels.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
label_classes = list(mrrs.geometry) #it gives the list of all polygon

#taking one polygon (polygon of interest)
a = mrrs[300].minimum_rotated_rectangle

l = a.boundary
coords = [c for c in l.coords]
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
segments  ##it stores the all the side lines of the polygon ()
geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
geoseries_shapely.to_file('C:\\Users\\binda\\Downloads\\check\\300.shp')

#iterate the side lines and store it in segment_lists
segment_lists = []
for i, segment_list in enumerate(segments):
    segment_list = segments[i].length
    segment_lists.append(segment_list)
    sorted_segment = sorted(segment_lists)   
#segment_lists gives the list of distance of each lines of polygon of interest

#storing the value of segments_lists in a separate column for each
geoseries_shapely['length'] = segment_lists
g = geoseries_shapely
list(g)

#now we have a length of each lines and we are going to take out only 2 longest line among 4.
#only calculated for if case
#if g[0] is selected, it is obvious that the opposite side would be g[2]
#similarly if g[1] is selected, it is obvious that the opposite side would be g[3]
lineSegments= []
if (g[0].length>g[1].length and g[0].length>g[3].length):
    lineSegment = geoseries_shapely[0]
    lineSegments.append(lineSegment)
    lineSegments.append(geoseries_shapely[2])
    print('I am hero')
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

    
lineSegments  ##the line segment would store 2 longest line
centroid_all = []
distance_all = []

#taking the 1st line longest line that is lineSegments[0]
#finding centroid of all the polygon and calculating distance between line and that centroid
#gives a list of distances which is stored in distance_all
for i, class_id in enumerate(label_classes):
    centroid_one = label_classes[i].centroid
    centroid_all.append(centroid_one)
    distance_between_points = lineSegments[0].distance(centroid_all[i])
    distance_all.append(distance_between_points)


distance_all
sort_distance_min = sorted(distance_all)  #sorting distance in ascending order
list_distance = sort_distance_min[0:4]   #taking only 4 elements out of distance_all
max_of_min = max(list_distance)       
#taking maximum value out of 4 elements gives us one unique value which we are going to 
#compare with another opposite line
sort_distance_min
list_distance
max_of_min
##doing the similar method for another opposite line
lineSegments[1]
centroid_all1 = []
distance_all1 = []

for i, class_id in enumerate(label_classes):
    centroid_one1 = label_classes[i].centroid
    centroid_all1.append(centroid_one1)
    distance_between_points1 = lineSegments[1].distance(centroid_all1[i])
    distance_all1.append(distance_between_points1)


distance_all1
sort_distance_min1 = sorted(distance_all1)
list_distance1 = sort_distance_min1[0:4]
max_of_min1 = max(list_distance1)
#max_of_min1 gives the unique value which we are going to compare with the previous max_of_min
sort_distance_min1
max_of_min1
#if the first line's distance with 4th polygon's centroid is less, it is probable 
#it is nearer to other segments (another house roof) and is considered gutter
if max_of_min < max_of_min1:
    gutter = lineSegments[1]
    print(gutter)
else:
    gutter = lineSegments[0]

gutter_shapely = gpd.GeoSeries(gutter, crs=4326)
gutter_shapely.to_file('C:\\Users\\binda\\Downloads\\check\\gutter_300.shp')



gpd.GeoSeries([gutter, mrrs[3]]).plot(cmap='RdYlGn_r' )    

for i, class_id in enumerate(centroid_all):
    gpd.GeoSeries([gutter, mrrs]).plot(color='Red')
    
