# Deep Learning for Photovoltaic Potential - Evaluation of an Alternative Approach for Roof Segmentation and Orientation Determination
This project is the code used for completing the master's thesis. The project demonstrate an alternative approach for roof segmentation and orientation determination. The approach is based on a deep learning model that is trained on a dataset of an aerial images. 

This repository includes datasets for semantic segmentation of roof segments and rooflines (gables and gutters). The code includes the data preparation, training, and comparison of the different algorithms adapted for getting the correct gutter segment. The code also includes the evaluation of the different algorithms.
 
## How to get started
- Clone the repository
- Install the requirements
- Download the datasets
- Run the code
- Some codes need to be manually run in a specific order

## Steps involved
- Data preparation: creation of the mask labels according to the requirement. For this task, the separate mask label for the the roof segments and rooflines are created. The code for this task is in the script main.py. The excel file contains the vector label and the necessary roof segments, rooflines (gutter and gable) dataset were prepared using vector_labels_to_masks in main.py
- Second step includes the addition and subtraction (image processing) of the prepared dataset. For this task, the roof line (except gutter) and roof segments were added to get a dataset which contains the roof segments and the rooflines. The function add_images() performs the task of the data preparation in main.py.
- Splitting of Dataset: train_val_test_split in main.py performs the task of splitting the dataset into train, validation, and test dataset but the data are stored in txt or excel file. The txt or excel file is used splitting the dataset into train, validation, and test dataset which are performed by the function train_image, train_mask, val_image, val_mask, test_image and test_mask in main.py.
- Step 4 includes the training of the dataset in GPU. The datasets were ordered in order to fit the code. In this thesis, mainly 2 datasets were trained: one with 3 classes (roof, rooflines, and gutter) and another with 2 classes (background and gutter). The function model_training in main.py performs the task of training the dataset in GPU.
- Step 5 includes the evaluation of the trained model by creating a predicted dataset and creating a confusion matrix. evaluate_model_predictions function evaluates the model predictions and creates a confusion matrix in main.py.
- The visualization of the predicted dataset along with the ground truth and original image is performed by the function visualize_top_median_bottom_predictions_and_ground_truth in main.py.
- Post processing is performed in the next step. remove_bg function in main.py performs the task of removing the rooflines from the predicted dataset. This helps to get the separate roof segments free from rooflines.
- Since the predicted dataset are not georeferenced, they are georeferenced in the next step. The function create_geotiff in main.py performs the task of georeferencing the predicted dataset with the original image dataset by using its corresponding image id (name).
- In nearest_gutter_algorithm.py the function orientation is used to get the orientation of the roof segments. The datasets: ground truth roof segment, ground truth gutter, predicted roof segment, and predicted gutter are used. The mask were converted to vector using function rediction_raster_to_vector and all are stored to different gdf_labels. 
- In nearest_gutter_algorithm.py the function segment_generation2 takes the input as the gdf_labels and the orientation and returns the roof segments with the correct orientation. The gdf data are passed and undergoes different spatial operation to get the correct azimuth and orientation. The nearest_gutter_vs_farthest_roof.py plays a similar role but has used different algorithm to get the azimuth and orientation. We can play with the algorithm to get the best result.

## some extra py files
- coloring_roof.py contains script for merging shapefile and plot function for azimuth histogram
- manually_save_predictions.py includes the script for manually saving the predicted dataset
- The orientation error in this thesis were handled in different ways. The code for this task is in mean_orientation_error.py and mean_orientation_error1.py
- MOE_plot.py contains the best script for plotting the mean orientation error  by subtracting the error from 360 degree if the difference is greater than 180 degree.
- pv_potential.py contains the script for calculating the PV potential of the roof segments but is out of scope of this thesis.
- confusion_matrix_2_classes.py helps to get to know about the confusion matrix (not necessary for to run the code)
-longest_line_algorithm_vs_nga.py contains the script for comparing the orientation error of the two algorithms, i.e., the comparison of the orientation of the ground truth (nearest gutter algorithm) and the orientation of the predicted dataset (longest line algorithm).

### Prerequisites
Required packages are included in the requirements.txt.
 
### Installing
When running the code on Windows, packages fiona, gdal, and geopandas need to be installed from wheel files. The wheel files can be found here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
Code can be used with Python 3.9 by installing packages from requirements_python_39.txt

## Running the Model/Code
The whole pipeline can be run using main.py. The steps include:
1) Create roof superstructure masks from vector labels
2) Create roof segment masks from vector labels
3) Analyze the dataset
4) Evaluate annotation experiment and visualize results
5) Train model for semantic segmentation of superstructure 
6) Evaluate model and visualize results
7) Conduct PV potential assessment

The pipeline can be run using different input datasets:
1) initial labels (with inferior label quality)
2) reviewed labels (with enhanced label quality)

Settings should be defined in definitions.py

It is recommended to run parts of the pipeline seperately, e.g. when training the model for semantic segmentation of roof segments, or when optimizing the model parameters.
 
## Deployment
Built With [Python](https://www.python.org/) 3.7.0

 
## Authors
Author: Deepak Parajuli
Master Thesis: Deep Learning for Photovoltaic Potential - Evaluation of an Alternative
Approach for Roof Segmentation and Orientation Determination

## Supervisors
Prof. Dr. rer. nat. Christine Preisach, Hochschule Karlsruhe,
Msc. Sebastian Krapf, TUM

## Contributors
Deepak Parajuli,
Sebastian Krapf,
Fabian Netzler, 
Lukas Bogenrieder, 
Nils Kemmerzell

## Credits
This work would not have been possible without numerous python packages such as keras, segmentation models, shapely, geopandas, and more. See requirements.txt for packages used.

## License
1) This project is licensed under the LGPL License - see the LICENSE.md file for details

2) The use of Google Satelite Images is allowed under principles of fair use. Images can be used for non-commercial purposes such as news, blogs, educational, recreational, or instructional use. For information, see: https://about.google/brand-resource-center/products-and-services/geo-guidelines/

