# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:59:21 2022

@author: binda
"""



import os
import shutil
import csv

def train_image(csvfile, destination_folder):

    #csvfile = 'D:/RID-master/RID-master/venv/segmentation_model_data/train_filenames_1_6_classes.csv'
#source_folder = 'D:/RID-master/RID-master/venv/data/augment_image_outputjhhj/'
    #destination_folder = 'D:/RID-master/RID-master/venv/data/csv_datasets/x_train_folder/'
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)

    with open(csvfile,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
    #header = next(reader)
    #print(header[0])

        for row in (reader):
            source = row[2]
            destination = destination_folder + os.path.basename(source)
            #print(destination)
      
        
            if os.path.isfile(source):
                shutil.copy(source, destination)
            #print('Moved: ', row)

                
def test_image(csvfile, destination_folder):

     #csvfile = 'D:/RID-master/RID-master/venv/segmentation_model_data/train_filenames_1_6_classes.csv'
 #source_folder = 'D:/RID-master/RID-master/venv/data/augment_image_outputjhhj/'
     #destination_folder = 'D:/RID-master/RID-master/venv/data/csv_datasets/x_train_folder/'
     if not os.path.isdir(destination_folder):
         os.mkdir(destination_folder)

     with open(csvfile,'r') as csvfile:
         reader = csv.reader(csvfile, delimiter=",")
     #header = next(reader)
     #print(header[0])

         for row in (reader):
             source = row[2]
             destination = destination_folder + os.path.basename(source)
             #print(destination)
       
         
             if os.path.isfile(source):
                 shutil.copy(source, destination)
             #print('Moved: ', row)

def train_mask(csvfile, destination_folder):

     #csvfile = 'D:/RID-master/RID-master/venv/segmentation_model_data/train_filenames_1_6_classes.csv'
 #source_folder = 'D:/RID-master/RID-master/venv/data/augment_image_outputjhhj/'
     #destination_folder = 'D:/RID-master/RID-master/venv/data/csv_datasets/x_train_folder/'
     if not os.path.isdir(destination_folder):
         os.mkdir(destination_folder)

     with open(csvfile,'r') as csvfile:
         reader = csv.reader(csvfile, delimiter=",")
     #header = next(reader)
     #print(header[0])

         for row in (reader):
             source = row[1]
             #print(source)
             destination = destination_folder + os.path.basename(source)
             #print(destination)
       
         
             if os.path.isfile(source):
                 shutil.copy(source, destination)
             #print('Moved: ', row)
             
def test_mask(csvfile, destination_folder):

     #csvfile = 'D:/RID-master/RID-master/venv/segmentation_model_data/train_filenames_1_6_classes.csv'
 #source_folder = 'D:/RID-master/RID-master/venv/data/augment_image_outputjhhj/'
     #destination_folder = 'D:/RID-master/RID-master/venv/data/csv_datasets/x_train_folder/'
     if not os.path.isdir(destination_folder):
         os.mkdir(destination_folder)

     with open(csvfile,'r') as csvfile:
         reader = csv.reader(csvfile, delimiter=",")
     #header = next(reader)
     #print(header[0])

         for row in (reader):
             source = row[1]
             #print(source)
             destination = destination_folder + os.path.basename(source)
             #print(destination)
       
         
             if os.path.isfile(source):
                 shutil.copy(source, destination)
             #print('Moved: ', row)