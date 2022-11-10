__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2021, "
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import os
import pickle
import cv2
import numpy as np
import albumentations as A

from mask_generation \
    import vector_labels_to_masks,vector_labels_to_masks_try, vector_labels_to_masks_gable,vector_labels_to_masks_gutter, vector_labels_to_masks_roofline, train_val_test_split, import_vector_labels
from add_images \
    import add_images
from augmentation \
    import augment, augment_mask
from copy_from_csv \
    import train_image, test_image, train_mask, test_mask
from conversion_raster_to_polygon \
    import input_output
from dataset_analysis \
    import mask_pixel_per_image, class_distribution, visualize_class_distribution
from annotation_experiment_evaluation \
    import evaluate_annotation_experiment, \
    visualize_annotation_experiment_confusion_matrix, \
    visualize_annotation_experiment_box_plot
from model_training \
    import model_training, get_datasets, get_test_dataset
from model_evaluation \
    import create_filter_dataset, evaluate_model_predictions, visualize_prediction_confusion_matrix, \
    visualize_prediction_mean_IoUs_as_box_plots, normalize_confusion_matrix_by_rows, \
    visualize_top_median_bottom_predictions_and_ground_truth, calculate_top_median_bottom_5,\
    df_IoU_from_confusion_matrix, model_load, save_prediction_masks
from utils \
    import get_image_gdf_in_directory, geotif_to_png
from visualization import visualization_annotation_agreement
from pv_potential import pv_potential_analysis
from mean_orientation_error import orientation_error
from nearest_gutter_algorithm import remove_bg, create_geotiff



### Define paths
from definitions import \
    DATA_DIR_ANNOTATION_EXPERIMENT, \
    DIR_IMAGES_GEOTIFF, \
    DIR_IMAGES_PNG, \
    OUTPUT_AUGMENT, \
    DIR_MASKS_SUPERSTRUCTURES, \
    DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT, \
    DIR_MASKS_SEGMENTS, \
    DIR_MASKS_SEGMENTS_TRY, \
    OUTPUT_AUGMENT_MASK, \
    DIR_MASKS_SEGMENTS_GABLE, \
    DIR_MASKS_SEGMENTS_GUTTER, \
    DIR_MASKS_SEGMENTS_ROOFLINE, \
    DIR_SEGMENTATION_MODEL_DATA, \
    DIR_RESULTS_TRAINING, \
    DIR_MASK_FILES, \
    DIR_PREDICTIONS, \
    FILE_VECTOR_LABELS_SUPERSTRUCTURES, \
    ROOF_FOLDER, \
    GABLE_FOLDER, \
    GUTTER_FOLDER, \
    OUTPUT_FOLDER, \
    SHP_OUTPUT_ROOFLINE, \
    FILE_VECTOR_LABELS_SEGMENTS, \
    FILE_VECTOR_LABELS_PV_AREAS, \
    FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT, \
    VAL_DATA_CENTER_POINTS,\
    IMAGE_SHAPE, \
    MODEL_NAME, \
    MODEL_TYPE, \
    DATA_VERSION, \
    BACKBONE, \
    CSV_FILE_TRAIN, \
    CSV_FILE_TEST, \
    DIR_TRAIN_FOLDER, \
    OUTPUT_FOLDER_ADD, \
    LABEL_CLASSES_ROOFLINE, \
    DIR_TRAIN_FOLDER3, \
    DIR_TEST_FOLDER3,DIR_TRAIN_M_FOLDER3, DIR_TEST_M_FOLDER3, \
    label_classes_superstructures_annotation_experiment1, \
    DIR_PREDICTIONS1, DIR_IMAGES_GEOTIFF_TRASH, DIR_CREATE_AND_DELETE

### Define labeling classes
from definitions import \
    LABEL_CLASSES_SUPERSTRUCTURES,\
    LABEL_CLASSES_SUPERSTRUCTURES1, \
    LABEL_CLASSES_GABLE,\
    LABEL_CLASSES_SEGMENTS, \
    LABEL_CLASSES_SEGMENTS_TRY, \
    LABEL_CLASSES_PV_AREAS
    

########################################################################################################################
### Import images
########################################################################################################################
# initialize png images, if pngs do not exist.
geotif_to_png(DIR_IMAGES_GEOTIFF, DIR_IMAGES_PNG)

# Get ids of all images in geotiff image folder
image_id_list = [id[:-4] for id in os.listdir(DIR_IMAGES_GEOTIFF) if id[-4:] == '.tif']
gdf_images = get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF)


# import labels from annotation experiment
gdf_test_labels = import_vector_labels(
    FILE_VECTOR_LABELS_ANNOTATION_EXPERIMENT,
    'superstructures',
    LABEL_CLASSES_SUPERSTRUCTURES
)


########################################################################################################################
### 1) Create roof superstructure masks from vector labels
########################################################################################################################
gdf_labels_superstructure = vector_labels_to_masks(
    FILE_VECTOR_LABELS_SUPERSTRUCTURES,
    DIR_MASKS_SUPERSTRUCTURES,
    'superstructures',
    LABEL_CLASSES_SUPERSTRUCTURES,
    gdf_images,
    filter=False
)


########################################################################################################################
### 2) Create roof segment masks from vector labels
########################################################################################################################
gdf_labels_segments = vector_labels_to_masks(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS,
    'segments',
    LABEL_CLASSES_SEGMENTS,
    gdf_images,
    filter=False
)

gdf_labels_try = vector_labels_to_masks_try(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS_TRY,
    'segments_try',
    LABEL_CLASSES_SEGMENTS_TRY,
    gdf_images,
    filter=False
)
"""

"""
#converts the gable (csv data) into a mask label
gdf_labels_segments_gable = vector_labels_to_masks_gable(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS_GABLE,
    'segments_gable',
    LABEL_CLASSES_SEGMENTS_TRY,
    gdf_images,
    filter=False
)

#converts the gutter (csv data) into a mask label
gdf_labels_segments_gutter = vector_labels_to_masks_gutter(
    FILE_VECTOR_LABELS_PV_AREAS,
    DIR_MASKS_SEGMENTS_GUTTER,
    'segments_gutter',
    LABEL_CLASSES_SEGMENTS_TRY,
    gdf_images,
    filter=False
)

#converts the roofline (csv data) into a mask label
gdf_labels_roofline = vector_labels_to_masks_roofline(
    FILE_VECTOR_LABELS_SEGMENTS,
    DIR_MASKS_SEGMENTS_ROOFLINE,
    'roofline',
    LABEL_CLASSES_SEGMENTS_TRY,
    gdf_images,
    filter=False
    )

#########################################
### 3) splitting of the train_validation data so that there is no any overlap between the train, validation
#and test dataset
#########################################
train_val_test_split(
    gdf_test_labels,
    gdf_images,
    VAL_DATA_CENTER_POINTS,
    LABEL_CLASSES_SEGMENTS,
    DIR_IMAGES_PNG,
    DIR_MASKS_SEGMENTS,
    DIR_SEGMENTATION_MODEL_DATA
)


#add two images and save it into a folder for generating different kind of masks
#eg: combination of rooflines, roof, gutters or gables for different study purpose
### Step 4) Ignore it if you already have required mask of your purpose
add_images(
    DIR_MASKS_SEGMENTS_TRY,
    DIR_MASKS_SEGMENTS_ROOFLINE,
    DIR_MASKS_SEGMENTS_GABLE,
    OUTPUT_FOLDER_ADD
    )

"""
#commented augment_mask as it is not necessary for our purpose
augment_mask(
    DIR_MASKS_SEGMENTS_ROOFLINE_ADD,
    OUTPUT_AUGMENT_ADD
    )

#for mixed image
augment_mask(
    OUTPUT_AUGMENT_ROOF_GABLE,
    DIR_ROOFL_GABLE
    )
"""
### Step 5) 
#copy the the image files from the numbers using csv file or text file created in step 3
#example: if the csv/txt file has values: 1,2,3,4,5 this script will copy the images with maskid
#1, 2, 3, 4, 5 from the image folder to another empty folder
train_image(
    CSV_FILE_TRAIN,
    DIR_TRAIN_FOLDER3
    )

test_image(
    CSV_FILE_TEST,
    DIR_TEST_FOLDER3
    )

train_mask(
    CSV_FILE_TRAIN,
    DIR_TRAIN_M_FOLDER3
    )

test_mask(
    CSV_FILE_TEST,
    DIR_TEST_M_FOLDER3
    )

"""
########################################################################################################################
### Step 6) Analyze the dataset
##removed for debbuging purpose and not important in our study
########################################################################################################################
# calculate the pixel share of the classes for superstructure and segment dataset
class_share_percent_superstructures = mask_pixel_per_image(
    DIR_MASKS_SUPERSTRUCTURES,
    LABEL_CLASSES_SUPERSTRUCTURES,
    IMAGE_SHAPE
)


class_share_percent_segments = mask_pixel_per_image(
    DIR_MASKS_SEGMENTS,
    LABEL_CLASSES_SEGMENTS,
    IMAGE_SHAPE
)
#pixel share of the roof line
class_share_percent_superstructures = mask_pixel_per_image(
    DIR_MASKS_SEGMENTS_ROOFLINE,
    LABEL_CLASSES_ROOFLINE,
    IMAGE_SHAPE
)
"""

"""
### step 7)
# calculate the number of labels and the labeled area for superstructure and segment dataset
label_class_count_superstructures, label_area_count_superstructures = class_distribution(
    gdf_labels_superstructure, LABEL_CLASSES_SUPERSTRUCTURES
)

label_class_count_segments, label_area_count_segments = class_distribution(
    gdf_labels_segments, LABEL_CLASSES_SEGMENTS
)


#visualize_class_distribution(LABEL_CLASSES_SUPERSTRUCTURES)



########################################################################################################################
### 8) Evaluate annotation experiment and visualize results
########################################################################################################################
# Evaluation annotation agreement of superstructure labels. This takes long time to compute.
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!

if os.path.isfile('data\\res_annotation_experiment.pkl'):
    with open('data\\res_annotation_experiment.pkl', 'rb') as f:
        [CM_AE_all, CM_AE_list, CM_AE_class_agnostic_all, CM_AE_class_agnostic_list] = pickle.load(f)
    # generate dataframes with all class specific IoUs
    df_IoU_AE = df_IoU_from_confusion_matrix(CM_AE_list, LABEL_CLASSES_SUPERSTRUCTURES)
    df_IoU_AE_class_agnostic = df_IoU_from_confusion_matrix(CM_AE_class_agnostic_list, ['label_class', 'background'])
else:
    image_id_list_annotation_experiment = os.listdir(DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT)

    df_IoU_AE, CM_AE_all, CM_AE_list, df_IoU_AE_class_agnostic, CM_AE_class_agnostic_all, CM_AE_class_agnostic_list =\
        evaluate_annotation_experiment(
            LABEL_CLASSES_SUPERSTRUCTURES,
            DIR_MASKS_SUPERSTRUCTURES_ANNOTATION_EXPERIMENT,
            image_id_list_annotation_experiment
        )
"""
"""
# visualize an example of two annotators labels

visualization_annotation_agreement(
    gdf_test_labels,
    LABEL_CLASSES_SUPERSTRUCTURES,
    annotator_ids=[1, 3],
    building_id=[5]
)

# visualize class specific annotation agreement as box plot
visualize_annotation_experiment_box_plot(df_IoU_AE, df_IoU_AE_class_agnostic, LABEL_CLASSES_SUPERSTRUCTURES)

# calculate normalized confusion matrix
CM_AE_all_normalized = normalize_confusion_matrix_by_rows(CM_AE_all)
visualize_annotation_experiment_confusion_matrix(CM_AE_all_normalized, LABEL_CLASSES_SUPERSTRUCTURES.values())
"""

# Evaluation annotation agreement of roof outline. This takes long time to compute
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!

"""
#not important for this study
if os.path.isfile('data\\res_annotation_experiment_pv_areas.pkl'):
    with open('data\\res_annotation_experiment_pv_areas.pkl', 'rb') as f:
        [CM_AE_pv_area_all, CM_AE_pv_area_list] = pickle.load(f)
else:
    image_id_pv_area_list_annotation_experiment = os.listdir(DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT)
    _, _, CM_AE_pv_area_list, _, _, _ =\
        evaluate_annotation_experiment(
            LABEL_CLASSES_PV_AREAS,
            DIR_MASKS_PV_AREAS_ANNOTATION_EXPERIMENT,
            image_id_pv_area_list_annotation_experiment,
        )

"""

# ########################################################################################################################
# ### 5) Train model for semantic segmentation of superstructure - Make sure to use a GPU
###carry out model_training in GPU and get the .h5 model and store it inside DIR_RESULTS_TRAINING
# ########################################################################################################################
model = model_training(MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES, DIR_SEGMENTATION_MODEL_DATA, DIR_MASKS_SEGMENTS_TRY, DIR_RESULTS_TRAINING, IMAGE_SHAPE)
model.load_weights(DIR_RESULTS_TRAINING + '/' + MODEL_NAME + '.h5', by_name=True)
########################################################################################################################
### 6) Evaluate model and visualize results
########################################################################################################################

DIR_SEGMENTATION_MODEL_DATA = DIR_SEGMENTATION_MODEL_DATA # + '_3' # use validation split 3

# load model and datasets
model, preprocess_input = model_load(MODEL_NAME, MODEL_TYPE, BACKBONE, LABEL_CLASSES_SUPERSTRUCTURES1)
model.summary()  #get summary of the model

train_dataset, valid_dataset, test_dataset = get_datasets(DIR_SEGMENTATION_MODEL_DATA, DIR_MASK_FILES, DATA_VERSION,
                                               preprocess_input, label_classes_superstructures_annotation_experiment1, resize=None)
#LABEL_CLASSES_SUPERSTRUCTURES1
#add the test txt file that is created in step 3) into the directory filenames_annotation_experiment
dir_mask_files_test = os.path.join(DIR_SEGMENTATION_MODEL_DATA, 'filenames_annotation_experiment')

#add all the the original image and masked image that is created from txt file in step 5) into a folder 
#D:\RID-master\RID-master\raster_data_annotation_experiment in test and test_masks
#use the same LABEL_CLASSES values which is used in the training of the model. Ex: use 2 or 3 classes according to requirement
test_dataset = get_test_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, dir_mask_files_test, 'annotation_experiment',
                                preprocess_input,
                                LABEL_CLASSES_SUPERSTRUCTURES1.values())

filter_dataset = create_filter_dataset(DATA_DIR_ANNOTATION_EXPERIMENT, dir_mask_files_test, 'annotation_experiment',
                                       LABEL_CLASSES_SUPERSTRUCTURES1, preprocess_input)

filter_dataset = None
# Evaluation of model.  This takes long time top compute on CPU.
# Check if results of evaluation are already saved as pkl file.
# Important: change the pkl filename when evaluating multiple models!

results_path = os.path.join('data\\res_model_predictions', 'res_model_predictions_UNet_2.pkl')
if os.path.isfile(results_path):
    with open(results_path, 'rb') as f:
        [CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list] = pickle.load(f)
    # generate dataframes with all class specific IoUs
    df_IoUs = df_IoU_from_confusion_matrix(CM_list, LABEL_CLASSES_SUPERSTRUCTURES1)
    df_IoU_class_agnostic = df_IoU_from_confusion_matrix(CM_class_agnostic_list, ['label_class', 'background'])
else:
    df_IoUs, CM_all, CM_list, df_IoU_class_agnostic, CM_class_agnostic_all, CM_class_agnostic_list = \
        evaluate_model_predictions(
            model,
            test_dataset,
            filter_dataset,
            LABEL_CLASSES_SUPERSTRUCTURES1,
            filter_center_roof=False
        )
    results_path = os.path.join('data\\res_model_predictions', 'res_model_predictions_UNet_2.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump([CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list], f)
with open(results_path, 'rb') as f:
    [CM_all, CM_list, CM_class_agnostic_all, CM_class_agnostic_list] = pickle.load(f)
# calculate normalized confusion matrix
CM_all_normalized = normalize_confusion_matrix_by_rows(CM_all)
visualize_prediction_confusion_matrix(CM_all_normalized, LABEL_CLASSES_SUPERSTRUCTURES1.values())

# not important in this section
""""
visualize_prediction_mean_IoUs_as_box_plots(
    df_IoUs,
    df_IoU_class_agnostic,
    df_IoU_AE,
    df_IoU_AE_class_agnostic,
    CM_AE_pv_area_list,
    LABEL_CLASSES_SUPERSTRUCTURES1.values()
)
"""

# visualized six images, two good, two medium and two bad predictions
id_top_5, id_median_5, id_bottom_5 = calculate_top_median_bottom_5(CM_list)

visualize_top_median_bottom_predictions_and_ground_truth(
    model,
    id_top_5,
    id_median_5,
    id_bottom_5,
    test_dataset,
    filter_dataset,
    LABEL_CLASSES_SUPERSTRUCTURES1
)


########################################################################################################################
### 7) Conduct PV Potential Assessment
########################################################################################################################
# 7) calculate predictions on the validation dataset and save the masks
save_prediction_masks(model, test_dataset, LABEL_CLASSES_SUPERSTRUCTURES1, DIR_PREDICTIONS1)
"""
# use prediction masks to calculate pv potential for 6 use cases
pv_potential_analysis()
"""
###Step 8) Postprocess the image according to requirement. This includes the mapping of the pixels and 
#replacing pixels using opencv
#input is the image received from save_prediction. Change the directory accordingly
in_put = "D:\\RID-master\\RID-master\\data\\masks_segments_gable"
output = "D:\\RID-master\\RID-master\\segmentation_model_data\\masks_superstructures_reviewed"   
remove_bg(in_put, output) 


###step 9) Create a geotiff image. The processed image in the step 8) is not georeferenced image and hence need to georeferenced
#where the 1st parameter is the georeferenced original image, 2nd is the mask from step 8) and 3rd parameter is the output folder where the 
#georeferenced mask is stored
create_geotiff(
    DIR_IMAGES_GEOTIFF_TRASH,
    DIR_PREDICTIONS1,
    DIR_CREATE_AND_DELETE
)

#Further carry out nearest_gutter_algorithm.py script for further image processing of the predicted masks 
#refer readme file for further details
