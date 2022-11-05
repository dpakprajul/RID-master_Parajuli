# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:15:20 2022

@author: binda
"""
import os
import cv2
import numpy as np

def add_images(roof, gable,gutter, output):
    if not os.path.isdir(output):
        os.mkdir(output)
        
    files_roof = [geotif[:-4] for geotif in os.listdir(roof) if geotif[-4:] == '.png']
    files_gutter = [geotif[:-4] for geotif in os.listdir(gutter) if geotif[-4:] == '.png']
    files_gable = [geotif[:-4] for geotif in os.listdir(gable) if geotif[-4:] == '.png']
    files_png = [png[:-4] for png in os.listdir(output) if png[-4:] == '.png']
    missing_pngs_list = [geotif for geotif in files_roof if geotif not in files_png]
    #print(missing_pngs_list)
    for i, img in enumerate(missing_pngs_list):
        
        png_file_path = os.path.join(output, img + '.png')
        
        roof_file_path = os.path.join(roof, img + '.png')
        gable_file_path = os.path.join(gable, img + '.png')
        gutter_file_pathj = os.path.join(gutter, img + '.png')
        

        roof_read = cv2.imread(roof_file_path, 0)
        gable_read = cv2.imread(gable_file_path, 0)
        gutter_read = cv2.imread(gutter_file_pathj, 0)
        
        #img_roof_sub1 = cv2.subtract(gutter_read, roof_read)
        gable_gutter = cv2.add(gable_read, gutter_read)
        
        image_roof_sub = cv2.subtract(gable_gutter, roof_read)
        
        #image_roof_gable_gutter = cv2.add(image_roof_gable, gutter_read)
        
        
        #this is for adding images #delete for using another approach
        #image_roof_gable = cv2.add(gable_read, roof_read)
        #image_roof_gable.crs = 4326
        
        image_roof_gable= cv2.add(image_roof_sub, gable_gutter)
        
        
        
        image_roof_gable[np.where(image_roof_gable == 1)] = 2
        image_roof_gable[np.where(image_roof_gable == 4)] = 1
        cv2.imwrite(png_file_path, image_roof_gable)
        # Show the image
        
  
      
    return
    
    
    
   
  