import os
import cv2
#import tensorflow as tf
import numpy as np
#import albumentations as A
#import segmentation_models as sm
#from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import sklearn.metrics

# create a customized colormap with 18 colors, the first being transparent for the background class
colormap_tab20 = plt.get_cmap("tab20")
color_transparent = np.array([1, 1, 1, 0]) # RGBA values for transparency
colors_tab20 = colormap_tab20(range(20))
colors_tab18 = np.insert(colors_tab20, 0, color_transparent, axis = 0) # add new color as first value of color list
colors_tab18 = np.delete(colors_tab18, 20, axis = 0) # remove last (21st) color from list
colors_tab18 = np.delete(colors_tab18, [15, 16], axis = 0) # remove grey colors from list
COLORMAP_TAB18 = ListedColormap(colors_tab18)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(32, 10)) #=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def visualize_image_and_labels(
    subplots_shape = None,
    figsize = None,
    figsize_per_image = True,
    dpi = 100,
    overlay_order = None,
    overlay_alpha = 0.5,
    pairwise_iou = None,
    show_legend = True,
    horizontal_legend = False,
    filepath_out = None,
    show_plot = True,
    fontsize_title = 12,
    fontsize_iou = 12,
    **images
):
    
    """
    Plot images in a row,
    optionally in a different, given configuration,
    optionally with adjusted figsize,
    optionally overlaying images transparently in a given order.
    
    subplots_shape:    Tuple (n_rows, n_cols) for subplots
    figsize:           Tuple (figsize_x, figsize_y), per default as absolute figsize
    figsize_per_image: Whether to interpret figsize as relative, i.e., per given image / subplot, or absolute
    dpi:               Just that. Can be used to scale the plot.
    overlay_order:     Array of length equal to number of images, one value per image:
                       0 to start new subplot for image
                       1 to overlay image transparently onto last started subplot
    overlay_alpha:     Alpha value for overlaying images
    pairwise_iou:      None or array. If given, length must be equal to number of images passed to the function.
                       Each element can be none (no IoU output) or a tuple of two numbers corresponding to
                       the positions of two images passed, then their IoU will be output below the subplot.
    show_legend:       Just that.
    horizontal_legend: If false, legend will be output vertically to the right, if true, horizontally to the bottom.
    filepath_out:      None, or path and filename of image to be saved, including file extension.
    show_plot:         Whether to show the figure.
    """
    
    # Determine number of subplots
    n = len(images)
    if overlay_order is not None:
        # If overlays are requested, subtract number of to-be-overlayed images from number of subplots
        n = n - sum(overlay_order)
    else:
        # If no overlays are requested, set up the overlay_order array to be 0 for each image (needed for further processing)
        overlay_order = [0] * n
    
    # Prepare values for the legend
    values = [i for i in range(0,18)]
    classes = [
        "bg  ",
        "n   ", "nne ", "ne  ", "ene ",
        "e   ", "ese ", "se  ", "sse ",
        "s   ", "ssw ", "sw  ", "wsw ",
        "w   ", "wnw ", "nw  ", "nnw ",
        "flat"
    ]
    
    # Set up configuration of subplots
    subplots_rows, subplots_cols = 1, n
    if subplots_shape is not None:
        subplots_rows, subplots_cols = subplots_shape
    
    # Determine figsize depending on which options were given, standard is 4
    figsize_x = 3.3 * subplots_cols
    figsize_y = 3.3 * subplots_rows
    if figsize is not None:
        figsize_x, figsize_y = figsize
        if figsize_per_image:
            figsize_x = figsize_x * subplots_cols
            figsize_y = figsize_y * subplots_rows
    
    iou_text_present = True if pairwise_iou is not None else False
    iou_text_in_layout = True
    
    fig = plt.figure(figsize = (figsize_x, figsize_y+0.16*(iou_text_present)), dpi = dpi, constrained_layout = True)
    
    # Subplots must be counted separately, because for an overlayed image, the number must not increase because no new subplot is created
    subplot_counter = 1
    
    ax = None
    
    # Create subplots iteratively
    for i, (name, image) in enumerate(images.items()):
        # If the current image is not to be overlayed, start a new subplot for it
        if overlay_order[i] == 0:
            ax = plt.subplot(subplots_rows, subplots_cols, subplot_counter)
            subplot_counter += 1
        
        plt.xticks([])
        plt.yticks([])
        #plt.title(' '.join(name.split('_')).title())
        titlestring = ' '.join(name.split('_'))
        titlestring = '_'.join(titlestring.split('.'))
        plt.title(titlestring, size = fontsize_title)
        
        im = plt.imshow(
            image,
            cmap = COLORMAP_TAB18, #plt.get_cmap("tab20"),
            vmin = 0,
            vmax = 17,
            interpolation = "none",
            alpha = (1 - (1 - overlay_alpha) * overlay_order[i]) # sets alpha to 1 unless current image is overlayed, then to overlay_alpha
        ) # cmap: tab20
        
        # Check whether to output an IoU under the current subplot
        if pairwise_iou is not None:
            if pairwise_iou[i] is not None:
                # pairwise_iou[i] is a tuple of two values indicating the truth and prediction index
                # in terms of the number of the corresponding images as passed to this function
                t_index, p_index = pairwise_iou[i]
                # get the corresponding images from all images that were passed as **images
                t = list(images.values())[t_index]
                p = list(images.values())[p_index]
                iou_micro = sklearn.metrics.jaccard_score(t.reshape(-1), p.reshape(-1), average = 'micro')
                iou_macro = sklearn.metrics.jaccard_score(t.reshape(-1), p.reshape(-1), average = 'macro')
                
                plt.text(
                    x = 0.5,
                    y = -0.08 - (2-subplots_rows)*0.01 - ((fontsize_iou-12)/12)*0.02,
                    #s = "IoU micro: {:.3f}, IoU macro: {:.3f}".format(iou_micro, iou_macro),
                    s = "Mi: {:.3f}, Ma: {:.3f}".format(iou_micro, iou_macro),
                    size = fontsize_iou,
                    horizontalalignment = "center",
                    transform = ax.transAxes,         # so that x and y are in axis coords,
                    in_layout = iou_text_in_layout    # iou text is ignored for layout calculations; set to True if iou text present in line that is not the most bottom one
                )
    
    if show_legend:
        # get the colors of the values, according to the colormap used by imshow
        colors = [im.cmap(im.norm(value)) for value in values]
        
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color = colors[i], label = classes[i] ) for i in range(len(values)) ]
        
        # put these patches as legend-handles into the legend

        if horizontal_legend:
            _, fy = figsize
            
            legend = fig.legend(
                handles = patches,
                loc = 8,
                frameon = False,
                ncol = 18,
                bbox_to_anchor = (0.4825, -0.24/subplots_rows + 0.06*(fy-3.8) - 0.02/subplots_rows*((fontsize_iou-12)/3)*iou_text_present*(1-iou_text_in_layout)), #-0.28/subplots_rows + 0.05*(fy-3.3)),
                columnspacing = -2.2,  # -2.1
                handlelength = 2,      #  2.2
                handleheight = 2.5     #  2.75
            )
            
            # move patches from left of text to above text
            for i, patch in enumerate(legend.get_patches()):
                patch.set_x(28) # x-position
                patch.set_y(18) # y-position
                # give first patch (white) a black frame
                if i == 0: patch.set_edgecolor([0, 0, 0, 1])
            
            # change font properties of texts
            for i, txt in enumerate(legend.get_texts()):
                #txt.set_ha("center") # horizontal alignment of text item
                txt.set_fontfamily("monospace")
                txt.set_fontsize(12) #12

        else:
            legend = fig.legend(handles = patches, loc = 7, frameon = False, bbox_to_anchor = (1 + 0.24/subplots_cols, 0.5))
            
            # give first patch (white) a black frame
            legend.get_patches()[0].set_edgecolor([0, 0, 0, 1])
    
    if filepath_out is not None: plt.savefig(filepath_out, format = filepath_out[-3:], dpi = dpi, bbox_inches = 'tight')
    
    if show_plot: plt.show()
    
    plt.close()


# helper function for data visualization
def visualize_image_and_labels_old(**images):
    
    """PLot images in one row."""
    
    n = len(images)
    plt.figure(figsize=(32, 10)) #=(16, 5))
    
    values = [i for i in range(0,18)]
    classes = [
        "background",
        "n", "nne", "ne", "ene",
        "e", "ese", "se", "sse",
        "s", "ssw", "sw", "wsw",
        "w", "wnw", "nw", "nnw",
        "flat"]
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        im = plt.imshow(image, cmap = plt.get_cmap("tab20"), vmin = 0, vmax = 17, interpolation = "none") # cmap: tab20
    
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color = colors[i], label=classes[i] ) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    plt.show()

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

####################################################
# DATASET AND DATALOADER
####################################################

class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    #CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
    #           'tree', 'signsymbol', 'fence', 'car', 
    #           'pedestrian', 'bicyclist', 'unlabelled']
    
    CLASSES = ["background",
               "n", "nne", "ne", "ene",
               "e", "ese", "se", "sse",
               "s", "ssw", "sw", "wsw",
               "w", "wnw", "nw", "nnw",
               "flat"]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.ids_annot = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_annot]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    def getitem_by_filename(self, filename):
        
        i = 0
        
        try:
            i = self.ids.index(filename)
        except:
            try:
                i = self.ids_annot.index(filename)
            except:
                raise ValueError("Could not find image or mask with given filename.")
        
        return self[i]
