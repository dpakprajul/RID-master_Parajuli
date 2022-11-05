# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 22:21:05 2022

@author: binda
"""

import os
#path = "D:\\RID-master\\RID-master\\data\\predicted_images_gutter"

#with open('D:\\trash\\myfile.txt', 'w') as file:
    #file.writelines([f for f in os.listdir(path)])
    
file_path = 'D:\\RID-master\\RID-master\\data\\predicted_images_gutter'
with open("D:\\trash\\myfile.txt", mode='w', newline='') as fp:
    for file in os.listdir(file_path):
        
        fp.write(str(file) + os.linesep)
        
import shutil

src = 'C:\\Users\\binda\\Downloads\\geotiff_try'
dst = 'C:\\Users\\binda\\Downloads\\geotiff_try\\test'

os.makedirs(dst, exist_ok=True)  # create destination folder if not exists

filenames = open("D:\\trash\\myfile.txt").read().split('\n')

for name in filenames:
    if name:  # to skip empty lines
        fullpath = os.path.join(src, name)
        if os.path.exists(fullpath):
            print('coping:', fullpath)
            shutil.copy(fullpath, dst)
        else:
            print('SKIPING:', fullpath)