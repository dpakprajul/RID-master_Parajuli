# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:50:10 2022

@author: binda
"""


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
    
    list3 = []
    list4 = []
    for i in range(len(az_list1)):
        
            #find the smallest value between a and b
           
            if abs(az_list1[i]-az_list2[i])>=180:
               angle_between = abs(az_list1[i]-az_list2[i])
               resulting_angle = 360-angle_between
               list3.append(resulting_angle)
               list4.append(resulting_angle**2)
            else:
                list3.append(abs(az_list1[i]-az_list2[i]))
                list4.append((abs(az_list1[i]-az_list2[i]))**2)
               
    plt.hist(list3, bins=40)
    plt.ylabel('Number of Roof Segments')
    plt.xlabel('Differences of two corresponding azimuth')
    plt.grid(color='0.5', linestyle='--', linewidth=1)
    plt.savefig('distribution of difference',dpi=2000)
         
                
        
    sum(list4)

    #find the mean of the list3
    sum(list4)/len(list4)

    #find the square root of the mean of the list3
    
    print(math.sqrt(sum(list4)/len(list4)))

    #find the standard deviation of the list3
    #import statistics
    #statistics.stdev(list3)
    

orientation_error()
