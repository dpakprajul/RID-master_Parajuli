# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 21:54:15 2022

@author: binda
"""

##buffer_approach

from pyproj import Geod
import geopandas as gpd
import heapq
import shapely
import pickle
import numpy
from shapely.geometry import Point, Polygon, MultiPoint, LineString
import math
import matplotlib.pyplot as plt
#from centerline_and_rooflien import segment_azimuth
import numpy as np
infile = open('C:\\Users\\binda\\Downloads\\check\\shape_file_original\\mydata.pkl', 'rb')
mrrs = pickle.load(infile)
mrrs = mrrs[0:1000]


label_classes = list(mrrs.geometry) 
segments = []

min_rot_each = mrrs[995].minimum_rotated_rectangle
boundary = min_rot_each.boundary
coords = [c for c in boundary.coords]
segments = [shapely.geometry.LineString([a,b]) for a,b in zip(coords, coords[1:])]
segments
geoseries_shapely = gpd.GeoSeries(segments, crs=4326)
save_directory = "C:\\Users\\binda\\Downloads\\check\\segment\\latest_segment.shp"
geoseries_shapely.to_file(save_directory)


segment_lists = []
for seg, segment_list in enumerate(segments):
    segment_list = segments[seg].length
    segment_lists.append(segment_list)
segment_lists
geoseries_shapely['length'] = segment_lists
g = geoseries_shapely


#calculate azimuth
def segment_azimuth(segment):
    x1, y1 = segment.coords[0]
    x2, y2 = segment.coords[1]
    azimuth = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return azimuth

#determine orientation according to azimuth in 16 directions 'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'flat'
def segment_orientation(azimuth):
    if azimuth < 0:
        azimuth += 360
    if azimuth >= 337.5 or azimuth < 22.5:
        return 'N'
    elif azimuth >= 22.5 and azimuth < 67.5:
        return 'NE'
    elif azimuth >= 67.5 and azimuth < 112.5:
        return 'E'
    elif azimuth >= 112.5 and azimuth < 157.5:
        return 'SE'
    elif azimuth >= 157.5 and azimuth < 202.5:
        return 'S'
    elif azimuth >= 202.5 and azimuth < 247.5:
        return 'SW'
    elif azimuth >= 247.5 and azimuth < 292.5:
        return 'W'
    elif azimuth >= 292.5 and azimuth < 337.5:
        return 'NW'
    else:
        return 'flat'


#import shapefile
sf = shapefile.Reader("C:\\Users\\binda\\Downloads\\check\\segment\\latest_segment.shp")

# create empty list to store azimuths
azimuths = []

# loop through each segment
for i in range(len(sf.shapes())):
    # get the x and y coordinates of the segment
    x = sf.shapes()[i].points[0][0]
    y = sf.shapes()[i].points[0][1]
    x2 = sf.shapes()[i].points[1][0]
    y2 = sf.shapes()[i].points[1][1]
    # calculate the azimuth
    azimuth = math.degrees(math.atan2(y2 - y, x2 - x))
    # append the azimuth to the list
    azimuths.append(azimuth)

# create empty list to store orientations
orientations = []

# loop through each azimuth
for azimuth in azimuths:
    # determine the orientation
    orientation = segment_orientation(azimuth)
    # append the orientation to the list
    orientations.append(orientation)

# create a new field in the shapefile
sf.field('orientation', 'C', '40')

# loop through each segment
for i in range(len(sf.shapes())):
    # get the orientation
    orientation = orientations[i]
    # write the orientation to the shapefile
    sf.record(orientation)

# save the shapefile
sf.save("C:\\Users\\binda\\Downloads\\check\\segment\\latest_segment.shp")



