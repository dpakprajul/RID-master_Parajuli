# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 20:44:48 2022

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
df_labels= gpd.read_file("C:\\Users\\binda\\Downloads\\download (3)\\layers\\POLYGON.shp")
gdf_labels = gpd.GeoDataFrame(df_labels, geometry='geometry')


geom1 = gdf_labels.geometry[7]
geom2 = gdf_labels.geometry[5]
#geom3 = gdf_labels.geometry[6]
buffer1 =  geom1.buffer(0.000002, cap_style = 3)
buffer2 = geom2.buffer(0.000002, cap_style = 3)
buffer = [buffer1, buffer2]
gpd.GeoSeries(buffer).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')
points = buffer1.boundary.intersection(buffer2.boundary) # multipoint
points = list(points) # point list
coords = [c for c in points[0].coords]
coords1 = [c for c in points[1].coords]
coords = Point(coords)
coords1 = Point(coords1)
print(coords)
lineString = LineString([coords, coords1])
lineString
geom_line = [buffer1,buffer2,lineString]
gpd.GeoSeries(geom_line).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')
mergedpoly = buffer1.union(buffer2)
mergedpoly
#buffer = mergedpoly.buffer(0.000002, cap_style = 3)
#buffer
mergepoly_1 = [mergedpoly,lineString]
gpd.GeoSeries(mergepoly_1).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')


geom1

geom = [geom1,geom2]

gpd.GeoSeries(geom).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')
points = geom1.boundary.intersection(geom2.boundary) # multipoint
points = list(points) # point list

print(points)

coords = [c for c in points[0].coords]
coords1 = [c for c in points[1].coords]
coords = Point(coords)
coords1 = Point(coords1)
print(coords)

lineString = LineString([coords, coords1])
lineString
geom_line = [geom1,geom2,lineString]
gpd.GeoSeries(geom_line).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')

mergedpoly = geom1.union(geom2)
mergedpoly
#buffer = mergedpoly.buffer(0.000002, cap_style = 3)
#buffer
mergepoly_1 = [mergedpoly,lineString]
gpd.GeoSeries(mergepoly_1).plot(color=['RED', 'BLUE','GREEN'], alpha=.2, edgecolor='black')


##last not working
geometry_list = []
for i, geometries in enumerate(gdf_labels.geometry):
    #print(gdf_labels)
    geom = gdf_labels.geometry[i]
    geometry_list.append(geom)
buffer =[]
for i, g in enumerate(geometry_list):
    buffer1 =  geometry_list[i].buffer(0.000002, cap_style = 3)
    buffer.append(buffer1)
  
buffer = buffer[0:2]
    
geo= gpd.GeoSeries(buffer, crs=4326)
plt.figure(figsize=((10,8)))
geo.plot()

