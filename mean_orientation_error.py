# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:06:51 2022

@author: binda
"""

#take a txt file from a directory and read the elements in the file
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def listToStringWithoutBrackets(list1):
    return str(list1).replace('[','').replace(']','')

def orientation_error():
#take a txt file from a C:\Users\binda\Downloads\check\azimuth_list az_list1
#and read the elements in the file
    with open(r'C:\\Users\\binda\\Downloads\\check\\azimuth_list\\az_list1.txt') as f:
        az_list1 = f.read().splitlines()

#take a txt file from a C:\Users\binda\Downloads\check\azimuth_list az_list2
#and read the elements in the file
    with open(r'C:\\Users\\binda\\Downloads\\check\\azimuth_list\\az_list2.txt') as f:
        az_list2 = f.read().splitlines()

#convert az_list1 to pandas series
    az_list1 = pd.Series(az_list1)

#convert az_list2 to pandas series
    az_list2 = pd.Series(az_list2)



#take all the each elements from the 1st column
    az_list1 = az_list1[0]
    az_list2=az_list2[0]
    
    az_list1 = listToStringWithoutBrackets(az_list1)
    az_list2 = listToStringWithoutBrackets(az_list2)
    
    az_list1 = az_list1.split(",")
    az_list2 = az_list2.split(",")
    az_list1 = [float(i) for i in az_list1]
    az_list2 = [float(i) for i in az_list2]
    list_a=[]
    list_b=[]
    for i in range(len(az_list1)):
        if (az_list1[i]-az_list2[i])<200 and (az_list1[i]-az_list2[i])>-200:
            list_a.append(az_list1[i])
            list_b.append(az_list2[i])
            
        
    
            #make a correlation plot with outliers
    plt.scatter(list_a, list_b, color='blue', marker='o', s=5)
    plt.xlabel('azimuth_list1')
    plt.ylabel('azimuth_list2')
    plt.title('Correlation plot with outliers')
    plt.show()

#save the plot
    plt.savefig('C:\\Users\\binda\\Downloads\\check\\azimuth_list\\correlation_plot_with_outliers.png')

    corr = np.corrcoef(list_a, list_b)[0,1]
    print(corr)



    list3 = []
    for i in range(len(az_list1)):
        if (az_list1[i]-az_list2[i])<150 and (az_list1[i]-az_list2[i])>-150:
            list3.append((az_list1[i]-az_list2[i])**2)
        else:
            pass
            
    #find the sum of the list3
    sum(list3)

    #find the mean of the list3
    sum(list3)/len(list3)

    #find the square root of the mean of the list3
    
    return math.sqrt(sum(list3)/len(list3))

    #find the standard deviation of the list3
    #import statistics
    #statistics.stdev(list3)










#






