__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"


import cv2
import os
import pickle
import sys

import numpy as np
from shapely.geometry import Point

##################################################
################ Define paths ####################
##################################################
# base directory



DIR_BASE = os.path.abspath(os.path.dirname(sys.argv[0]))
print ('full path =', DIR_BASE)

# data directory
DIR_DATA = DIR_BASE + "\\data"
print(DIR_DATA)

# training files directories
DIR_SEGMENTATION_MODEL_DATA = DIR_BASE + "\\" + "segmentation_model_data"
print(DIR_SEGMENTATION_MODEL_DATA)
# result directories
DIR_RESULTS_TRAINING = DIR_BASE + "\\results"
DIR_PREDICTIONS = DIR_BASE + "\\data" + "\\output_folder"
DIR_PREDICTIONS1 = DIR_BASE + "\\data" + "\\output_folder2"
# make paths if they do not exist
if not os.path.isdir(DIR_DATA): os.mkdir(DIR_DATA)
if not os.path.isdir(DIR_SEGMENTATION_MODEL_DATA): os.mkdir(DIR_SEGMENTATION_MODEL_DATA)
if not os.path.isdir(DIR_RESULTS_TRAINING): os.mkdir(DIR_RESULTS_TRAINING)
if not os.path.isdir(DIR_PREDICTIONS): os.mkdir(DIR_PREDICTIONS)
if not os.path.isdir(DIR_BASE + "\\plot"): os.mkdir(DIR_BASE + "\\plot")

# image directories
DIR_IMAGES_GEOTIFF = DIR_DATA + "\\images_roof_centered_geotiff"  # "images_annotation_experiment_geotiff" #
DIR_IMAGES_GEOTIFF_COPY = DIR_DATA + "\\images_roof_centered_geotiff_copy"
DIR_IMAGES_PNG = DIR_DATA + "\\images_roof_centered_png"  # images_annotation_experiment_png"
DIR_PREDICTED_IMAGES = DIR_DATA + "\\predicted_images"
DIR_PREDICTED_IMAGES1 = DIR_DATA + "\\predicted_images_gutter"
DIR_IMAGES_GEOTIFF_TRASH = "D:\\RID-master\\RID-master\\data\\gutter_image_trash"
PNG_TO_GEOTIFF = DIR_DATA + "\\predicted_geotiff"
PNG_TO_GEOTIFF1 = DIR_DATA + "\\predicted_geotiff_gutter"
OUTPUT_FROM_PREDICTION = DIR_DATA + "\\output_from_prediction"
OUTPUT_FROM_PREDICTION1 = DIR_DATA + "\\output_from_prediction_gutter"
# mask directories
DIR_MASKS_SUPERSTRUCTURES = DIR_DATA + "\\masks_superstructures_reviewed" #_initial"
DIR_MASKS_SEGMENTS = DIR_DATA + "\\masks_segments"
DIR_MASKS_SEGMENTS_TRY = DIR_DATA + "\\masks_segments_try"
DIR_MASKS_SEGMENTS_GABLE = DIR_DATA + "\\masks_segments_gable"
DIR_MASKS_SEGMENTS_GUTTER = DIR_DATA + "\\masks_segments_gutter"
DIR_MASKS_SEGMENTS_ROOFLINE = DIR_DATA + "\\masks_segments_roofline"
DIR_MASKS_SEGMENTS_ROOFLINE_ADD = DIR_DATA + "\\csv_datasets_v3" + "\\train_mask"
# annotation experiment directories
DATA_DIR_ANNOTATION_EXPERIMENT = DIR_BASE + '\\raster_data_annotation_experiment'
DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT = DIR_DATA + "\\masks_superstructures_annotation_experiment"
DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT = DIR_DATA + "\\masks_pv_areas_annotation_experiment"
# training files
DIR_MASK_FILES = DIR_SEGMENTATION_MODEL_DATA
# vector label files
FILE_VECTOR_LABELS_SUPERSTRUCTURES = "data\\" + "obstacles_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_SEGMENTS = "data\\" + "segments_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_PV_AREAS = "data\\" + "pv_areas_reviewed.csv" #_initial.csv" #
FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT = "data\\" + "obstacles_annotation_experiment.csv"

DIR_CREATE_AND_DELETE = DIR_DATA + "\\create_and_delete"
CSV_FILE_TRAIN = DIR_BASE + "\\" + "segmentation_model_data" + "\\train_filenames_1_6_classes.csv"
CSV_FILE_TEST = DIR_BASE + "\\" + "segmentation_model_data" + "\\test_filenames_1_6_classes.csv"


#deepak coded
ROOF_FOLDER= DIR_DATA + "\\images_roof_centered_geotiff"
GABLE_FOLDER = DIR_DATA + "\\masks_segments_gable"
GUTTER_FOLDER = DIR_DATA + "\\masks_segments_gutter"
OUTPUT_FOLDER = DIR_DATA + "\\output_folder"
OUTPUT_FOLDER_ADD = DIR_DATA + "\\output_folderadd"
SHP_OUTPUT = DIR_DATA + "\\output_shapefile"
SHP_OUTPUT_ROOFLINE = DIR_DATA + "\\output_shapefile_1"
OUTPUT_AUGMENT = DIR_DATA + "\\augment_image_output"
OUTPUT_AUGMENT_ADD = DIR_DATA + "\\augment_image_outputadd"
OUTPUT_AUGMENT_MASK = DIR_DATA + "\\augment_mask_output"
OUTPUT_AUGMENT_ROOF_GABLE = DIR_DATA + "\\roof_gable"
OUTPUT_AUGMENT_MASK_MIXED = DIR_DATA + "\\augment_mask_output_mix"
OUTPUT_AUGMENT_MASK_ROOFLINE = DIR_DATA + "\\augment_mask_output_roofline"
DIR_ROOFLINE_TESTED = DIR_DATA + "\\output_paper" + "\\roof_line_100_epoch"
DIR_ROOFL_GABLE =DIR_DATA + "\\augmented_roof_gable"
DIR_TRAIN_FOLDER = DIR_DATA + "\\csv_datasets_v2" + "\\train_images\\"
DIR_TEST_FOLDER = DIR_DATA + "\\csv_datasets_v2" + "\\test_images\\"
DIR_TRAIN_M_FOLDER = DIR_DATA + "\\csv_datasets_v2" + "\\train_mask\\"
DIR_TEST_M_FOLDER = DIR_DATA + "\\csv_datasets_v2" + "\\test_mask\\"

DIR_TRAIN_FOLDER3 = DIR_DATA + "\\csv_datasets_v3" + "\\train_images\\"
#print(DIR_TRAIN_FOLDER3)
DIR_TEST_FOLDER3 = DIR_DATA + "\\csv_datasets_v3" + "\\test_images\\"
DIR_TRAIN_M_FOLDER3 = DIR_DATA + "\\csv_datasets_v3" + "\\train_mask\\"
DIR_TEST_M_FOLDER3 = DIR_DATA + "\\csv_datasets_v3" + "\\test_mask\\"

COMPARISON_SEGMENT = DIR_DATA + "\\comparison" + "\\roof_segment"
COMPARISON_GUTTER = DIR_DATA + "\\comparison" + "\\gutter"
COMPARISON_ORIG = DIR_DATA + "\\comparison" + "\\original"
##################################################
########### Define class definition ##############
##################################################
# ## ALL labeled classes - Choose class definition
# label_classes_superstructures_all = ['background', 'unknown', 'window', 'ladder', 'shadow', 'chimney',
#                      'pvmodule', 'tree', 'dormer', 'balkony']
# label classes used in annotation experiment
label_classes_superstructures_annotation_experiment = ['pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow',
                                                       'tree', 'unknown'] #
label_classes_superstructures_annotation_experiment1 = ['roofline', 'roof'] #
## Label classes of segments - Choose class definition
label_classes_segments_6 = ['N', 'E', 'S', 'W']
label_classes_segments_10 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'flat']
#flat removed in 18 class for evaluation of the roof orientation
label_classes_segments_18 = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW','flat']
label_classes_segments_try = ['Roof']                

label_clases_pv_areas = ['background','pv_area']

label_classes_gable = ['Gabled', 'Flat']

label_classes_for_roofline = ['roof']


# decide here which classes to use to prepare the dataset!
LABEL_CLASSES_SUPERSTRUCTURES = dict(zip(np.arange(0, len(label_classes_superstructures_annotation_experiment)),
                                         label_classes_superstructures_annotation_experiment))
LABEL_CLASSES_SUPERSTRUCTURES1 = dict(zip(np.arange(0, len(label_classes_superstructures_annotation_experiment1)),
                                         label_classes_superstructures_annotation_experiment1))

LABEL_CLASSES_SEGMENTS = dict(zip(np.arange(0, len(label_classes_segments_18)), label_classes_segments_18))
LABEL_CLASSES_SEGMENTS_TRY = dict(zip(np.arange(0, len(label_classes_segments_try)), label_classes_segments_try))

LABEL_CLASSES_PV_AREAS = dict(zip(np.arange(0, len(label_classes_superstructures_annotation_experiment1)), label_classes_superstructures_annotation_experiment1))


LABEL_CLASSES_GABLE = dict(zip(np.arange(0, len(label_classes_gable)), label_classes_gable))
LABEL_CLASSES_ROOFLINE = dict(zip(np.arange(0, len(label_classes_for_roofline)), label_classes_for_roofline))
LABEL_CLASSES_6 = dict(zip(np.arange(0, len(label_classes_segments_6)), label_classes_segments_6))
# Manually define center points of validation data circles
north = Point([11.985659535136675, 48.41290587924208])
west = Point([11.975791189473085, 48.400038828407816])
east = Point([11.99794882677886, 48.39994299101589])
center_north = Point([11.987591755393657, 48.406794909515014])
center_south = Point([11.991633539397318, 48.40346489660213])

VAL_DATA_CENTER_POINTS = list([north, west, east, center_north, center_south])

# Coordinate systems
EPSG_METRIC = 25832

# Neural Network Parameters
# Neural Network Parameters

MODEL_NAME = 'UNet_2_initial'
MODEL_TYPE = 'UNet' # options are: 'Unet', 'FPN' or 'PSPNet'
BACKBONE = 'resnet34' #resnet34, efficientnetb2
DATA_VERSION = '1_6_classes'  # 2_rev, 3_rev, 4_initial ...




IMAGE_SHAPE = cv2.imread(DIR_IMAGES_GEOTIFF + '\\' + os.listdir(DIR_IMAGES_GEOTIFF)[0], 0).shape

############################################################
########### Look Up Table Technical Potential ##############
############################################################
lookup_path = os.path.abspath(os.path.join(DIR_BASE, 'data', 'df_technical_potential_lookup.pkl'))
# open lookup table
with open(lookup_path, 'rb') as f:
    df_technical_potential_LUT = pickle.load(f)
    

