# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:51:16 2022

@author: binda
"""
#confusion matrix between the ground truth and the prediction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import os



#confusion matrix between the ground truth and the prediction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model_evaluation \
    import create_filter_dataset, evaluate_model_predictions, visualize_prediction_confusion_matrix, \
    visualize_prediction_mean_IoUs_as_box_plots, normalize_confusion_matrix_by_rows, \
    visualize_top_median_bottom_predictions_and_ground_truth, calculate_top_median_bottom_5,\
    df_IoU_from_confusion_matrix, model_load, save_prediction_masks

#take all the ground truth images from the directory
y_true = [cv2.imread('D:\\RID-master\\RID-master\\data\\masks_segments_gable\\new\\' + name, 0) for name in os.listdir('D:\\RID-master\\RID-master\\data\\masks_segments_gable\\new\\')]
#take all the prediction images from the directory
y_pred = [cv2.imread('D:\\RID-master\\RID-master\\data\\output_folder2\\' + name, 0) for name in os.listdir('D:\\RID-master\\RID-master\\data\\output_folder2\\')]
#make a list of the names of the classes
class_names = ['gutter']

#convert y_true and y_pred to a numpy array
y_true = np.array(y_true)
y_pred = np.array(y_pred)
#make the above data to fit in the confusion matrix
y_true = y_true.reshape(-1)
y_pred = y_pred.reshape(-1)
#make a confusion matrix
cm = confusion_matrix(y_true, y_pred)
CM_all_normalized = normalize_confusion_matrix_by_rows(cm)
#make a confusion matrix
visualize_prediction_confusion_matrix(CM_all_normalized, class_names)
#save the heatmap as a png file
plt.savefig('D:\\RID-master\\RID-master\\plots\\confusion_matrix.png', format='png', dpi=300, bbox_inches='tight')



#from the confusion matrix, calculate the accuracy, precision, recall and f1 score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)
#calculate the precision
precision = precision_score(y_true, y_pred)
#calculate the recall
recall = recall_score(y_true, y_pred)
#calculate the f1 score
f1 = f1_score(y_true, y_pred)
