# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 20:21:20 2022

@author: binda
"""

import cv2
import numpy as np
from PIL import Image


#filename_mask = "D:\\RID-master\\RID-master\\data\\csv_datasets_v3\\val_masks\\val\\9.png"
filename_mask = "C:\\Users\\binda\\Downloads\\Image_output22.png"
png_file_path = "D:\\RID-master\\RID-master\\data\\csv_datasets_v3\\val_masks\\val\\9_output1.png"
roof_read = cv2.imread(filename_mask, 1)
img_gray = cv2.cvtColor(roof_read, cv2.COLOR_BGR2GRAY)
img_gray.flatten()
img_gray = img_gray/255
cv2.imwrite(png_file_path, img_gray)
