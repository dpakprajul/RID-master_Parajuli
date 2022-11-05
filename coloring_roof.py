# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 16:23:30 2022

@author: binda
"""

import plotly
import json
import numpy as np
import pandas as pd
import plotly.express as px
import geopandas as gpd
import pandas as pd
from shapely import speedups
import itertools
colorscales = px.colors.named_colorscales()
from matplotlib import *



from pathlib import Path
import pandas
import geopandas

folder = Path("C:\\Users\\binda\\Downloads\\try_azimuth")
shapefiles = folder.glob("*.shp")

gdf = pandas.concat([
   geopandas.read_file(str(shp))
   for shp in shapefiles
]).pipe(geopandas.GeoDataFrame)


gdf.to_file(folder / 'compiled.shp')


filenames = open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list1.txt").read()
filenames
merged = [j for i in filenames for j in i]
with open("C:\\Users\\binda\\Downloads\\check\\azimuth_list\\list3.txt", "w") as output:
    output.write(str(merged))



#take a csv file
file_vector_labels = "D:\\RID-master\\RID-master\\data\\pv_areas_reviewed.csv"
df_labels = pd.read_csv(file_vector_labels)
#plot the roof_type column
df_labels.roof_type.value_counts().plot(kind='bar')
plt.show()

#save the plot
plt.savefig('C:\\Users\\binda\\Downloads\\try_azimuth\\roof_type.png')


#take a csv file from 
file_vector_labels = "D:\\RID-master\\RID-master\\data\\segments_reviewed.csv"

#summarize the azimuth column and draw a historgram with a bin size of 30 degrees
df_labels = pd.read_csv(file_vector_labels)

#take a column azimuth
df_labels = df_labels[df_labels.azimuth.notna()]



#check if the data is negative. If it is, add 360 to the negative values
df_labels.loc[df_labels['azimuth'] < 0, 'azimuth'] += 360

#check if the data is greater than 360. If it is, subtract 360 from the values
df_labels.loc[df_labels['azimuth'] > 360, 'azimuth'] -= 360


#draw a histogram
df_labels.azimuth.hist(bins=22)
plt.xlabel('azimuth')
plt.ylabel('count')
plt.show()

#save a plot
plt.savefig('azimuth_histogram.png')


