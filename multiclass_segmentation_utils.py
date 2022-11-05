import os
import cv2
import tensorflow as tf
import numpy as np
import albumentations as A
import segmentation_models as sm
from tensorflow import keras
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

class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        # changed the following line from
        # "return batch" to "return tuple(batch)"
        # following advice on
        # https://github.com/qubvel/segmentation_models/issues/412
        # after model.fit threw following error:
        # ValueError: Layer model_4 expects 1 input(s), but it received 2 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, None, None, None) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(None, None, None, None) dtype=float32>]
        # note that this error did not occur on Google Colab using Tensorflow and Keras 2.7.0 (here: 2.5.0)
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


####################################################
# AUGMENTATION
####################################################

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation(random_grid_shuffle = True):
    
    train_transform = [
        
        #____ Scaling / Shifting / Cropping
        
        # removed horizontal flipping in order not to confuse the model about roof orientations
        ####A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        ####A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        ####A.RandomCrop(height=320, width=320, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=True),
        
        # Trying to replace the above 3 augmentations (ShiftScaleRotate, PadIfNeeded, and RandomCrop)
        # with a single one: Either CropAndPad (but will either only do center crops or crops that
        # change the aspect ratio of the image before rescaling to 256*256) or RandomResizedCrop.
        
        #A.CropAndPad(px = (-48, -16), sample_independently = True, p = 0.5),
        #A.RandomResizedCrop(width = 256, height = 256, scale = (0.5, 1), ratio = (1, 1), p = 1),
        
        #____ Other manipulations
        
        ####A.IAAAdditiveGaussianNoise(p=0.2),
        A.GaussNoise(p=0.2),
        
        # removed perspective in order not to confuse the model about roof orientations
        ####A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                ####A.RandomBrightness(p=1),
                A.RandomBrightnessContrast(contrast_limit = 0, p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                ####A.IAASharpen(p=1),
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                ####A.RandomContrast(p=1),
                A.RandomBrightnessContrast(brightness_limit = 0, p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        
        A.Lambda(mask=round_clip_0_1)
    ]
    
    if random_grid_shuffle:

        #_____ RandomGridShuffle

        # insert RandomGridShuffle in first position of the transform list
        train_transform.insert(
            0,
            A.OneOf(
                [
                    A.RandomGridShuffle(grid = (2, 2), p = 1.0),
                    A.RandomGridShuffle(grid = (3, 3), p = 1.0)
                ],
                p = 0.5
            )
        )
    
    return A.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #A.PadIfNeeded(384, 480)
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


####################################################
# MODEL SETUP AND TRAINING (OO)
####################################################

class Experiment:
    
    def __init__(
        self,
        data_dir,
        backbone,
        total_loss,
        batch_size,
        classes,
        lr = 1e-4,
        epochs = 40,
        experiment_dirpath = "",
        weights_in_filepath = None,
        save_weights_only = True,
        save_best_only = True,
        use_h5_format = False,
        backup_and_restore = False,
        random_grid_shuffle = True
    ):
        self.data_dir = data_dir
        self.backbone = backbone
        self.total_loss = total_loss
        self.batch_size = batch_size
        self.classes = classes
        self.lr = lr
        self.epochs = epochs
        self.experiment_dirpath = experiment_dirpath
        self.weights_in_filepath = weights_in_filepath
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.use_h5_format = use_h5_format
        self.backup_and_restore = backup_and_restore
        self.random_grid_shuffle = random_grid_shuffle
        
        self.backup_dirpath = os.path.join(self.experiment_dirpath, "backup")
        
        self.log_filepath = os.path.join(self.experiment_dirpath, "log.csv")
        
        if self.save_best_only:
            self.weights_out_filepath = os.path.join(experiment_dirpath, "best_model")
        else:
            self.weights_out_filepath = os.path.join(experiment_dirpath, "model_{epoch:02d}-{val_loss:.4f}")
        
        if self.use_h5_format:
            self.weights_out_filepath = "".join([self.weights_out_filepath, ".h5"])
        
        if self.backup_and_restore and not os.path.exists(self.backup_dirpath):
            os.makedirs(self.backup_dirpath)
        
        self.n_classes = len(self.classes)
        self.preprocess_input = sm .get_preprocessing(self.backbone)
        self.activation = "softmax"
        
        self.setup_model()
        self.setup_dataloaders()
        
        if self.weights_in_filepath is not None:
            self.model.load_weights(self.weights_in_filepath)
    
    def setup_model(self):
        # create model
        self.model = sm.Unet(self.backbone, classes = self.n_classes, activation = self.activation)

        # define optimizer
        self.optim = keras.optimizers.Adam(self.lr)

        self.metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimizer, loss and metrics
        self.model.compile(self.optim, self.total_loss, self.metrics)
    
    def setup_dataloaders(self):
        self.x_train_dir = os.path.join(self.data_dir, 'train')
        self.y_train_dir = os.path.join(self.data_dir, 'trainannot')

        self.x_valid_dir = os.path.join(self.data_dir, 'val')
        self.y_valid_dir = os.path.join(self.data_dir, 'valannot')

        #self.x_test_dir = os.path.join(self.data_dir, 'test')
        #self.y_test_dir = os.path.join(self.data_dir, 'testannot')
        
        # Dataset for train images
        self.train_dataset = Dataset(
            self.x_train_dir, 
            self.y_train_dir, 
            classes = self.classes, 
            augmentation = get_training_augmentation(random_grid_shuffle = self.random_grid_shuffle),
            preprocessing = get_preprocessing(self.preprocess_input),
        )

        # Dataset for validation images
        self.valid_dataset = Dataset(
            self.x_valid_dir, 
            self.y_valid_dir, 
            classes = self.classes, 
            #augmentation = get_validation_augmentation(), # only pads to reach 256*256 which is always the case anyway
            preprocessing = get_preprocessing(self.preprocess_input),
        )
        
        '''
        self.test_dataset = Dataset(
            self.x_test_dir, 
            self.y_test_dir, 
            classes = self.classes, 
            #augmentation = get_validation_augmentation(), # only pads to reach 256*256 which is always the case anyway
            preprocessing = get_preprocessing(self.preprocess_input)
        )
        '''

        self.train_dataloader = Dataloader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.valid_dataloader = Dataloader(self.valid_dataset, batch_size = 1, shuffle = False)
        #self.test_dataloader = Dataloader(self.test_dataset, batch_size = 1, shuffle = False)
        
        self.setup_test_dataset()
    
    def setup_test_dataset(self, dataset_dirpath = None):
        
        if dataset_dirpath == None: dataset_dirpath = self.data_dir
        
        self.x_test_dir = os.path.join(dataset_dirpath, 'test')
        self.y_test_dir = os.path.join(dataset_dirpath, 'testannot')
        
        self.test_dataset = Dataset(
            self.x_test_dir, 
            self.y_test_dir, 
            classes = self.classes, 
            #augmentation = get_validation_augmentation(), # only pads to reach 256*256 which is always the case anyway
            preprocessing = get_preprocessing(self.preprocess_input)
        )
        
        self.test_dataloader = Dataloader(self.test_dataset, batch_size = 1, shuffle = False)
    
    def fit_model(self):
        # check shapes for errors
        assert self.train_dataloader[0][0].shape == (self.batch_size, 256, 256, 3)
        assert self.train_dataloader[0][1].shape == (self.batch_size, 256, 256, self.n_classes)

        # define callbacks for learning rate scheduling and best checkpoints saving
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.weights_out_filepath, save_weights_only = self.save_weights_only, save_best_only = self.save_best_only, monitor = "val_loss", mode = "min"),
            keras.callbacks.ReduceLROnPlateau(),
            keras.callbacks.CSVLogger(self.log_filepath),
        ]
        
        if self.backup_and_restore:
            backup_callback = keras.callbacks.experimental.BackupAndRestore(backup_dir = self.backup_dirpath)
            callbacks.append(backup_callback)

        # train model
        self.history = self.model.fit(
            self.train_dataloader, 
            steps_per_epoch = len(self.train_dataloader), 
            epochs = self.epochs, 
            callbacks = callbacks, 
            validation_data = self.valid_dataloader, 
            validation_steps = len(self.valid_dataloader),
        )
        
    def plot_history(self):
        plt.figure(figsize=(15, 15))

        # Plot training & validation loss values
        plt.subplot(311)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])  
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Plot training & validation iou_score values
        plt.subplot(312)
        plt.plot(self.history.history['iou_score'])
        plt.plot(self.history.history['val_iou_score'])
        plt.title('Model iou-score') 
        plt.ylabel('iou_score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Plot training & validation f1_score values
        plt.subplot(313)
        plt.plot(self.history.history['f1-score'])
        plt.plot(self.history.history['val_f1-score'])
        plt.title('Model f1-score')
        plt.ylabel('f1-score')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.experiment_dirpath, "loss_iou_f1.png"), bbox_inches = "tight")
        plt.savefig(os.path.join(self.experiment_dirpath, "loss_iou_f1.pdf"), bbox_inches = "tight")
        
        plt.show()
    
    def test_model(self, test_weights_filepath = None, dataset_dirpath = None):
        
        # following was intended to load the weights of the best found model,
        # deactivated because not working with custom weights filenames.
        '''
        # if no other path is given for weights to be used, use own path
        if test_weights_filepath == None:
            test_weights_filepath = self.weights_out_filepath
        
        # load weights
        self.model.load_weights(test_weights_filepath)
        '''

        # load weights if given
        if test_weights_filepath is not None:
            self.model.load_weights(test_weights_filepath)
        
        # either use own test dataset / dataloader, or create dataset / dataloader for differing test dataset
        if dataset_dirpath == None:
            test_dataloader = self.test_dataloader
        else:
            x_test_dir = os.path.join(dataset_dirpath, 'test')
            y_test_dir = os.path.join(dataset_dirpath, 'testannot')
            
            test_dataset = Dataset(
                x_test_dir, 
                y_test_dir, 
                classes = self.classes, 
                #augmentation = get_validation_augmentation(), # only pads to reach 256*256 which is always the case anyway
                preprocessing = get_preprocessing(self.preprocess_input)
            )
            
            test_dataloader = Dataloader(test_dataset, batch_size = 1, shuffle = False)

        scores = self.model.evaluate(test_dataloader)

        print("Loss: {:.5}".format(scores[0]))
        for metric, value in zip(self.metrics, scores[1:]):
            print("mean {}: {:.5}".format(metric.__name__, value))
        
        #metric_names = ["loss"]
        #metric_names.append([metric.__name__ for metric in self.metrics])
        
        #return dict(zip(metric_names, scores))
        
        return scores
    
    def visualize_results(self, sample_ids = range(1, 10), test_weights_filepath = None, dataset_dirpath = None):
        # following was intended to load the weights of the best found model,
        # deactivated because not working with custom weights filenames.
        '''
        if test_weights_filepath == None:
            test_weights_filepath = self.weights_out_filepath
        
        # load weights
        self.model.load_weights(test_weights_filepath)
        '''
        
        if test_weights_filepath is not None:
            self.model.load_weights(test_weights_filepath)
        
        # either use own test dataset / dataloader, or create dataset / dataloader for differing test dataset
        if dataset_dirpath == None:
            test_dataset = self.test_dataset
        else:
            x_test_dir = os.path.join(dataset_dirpath, 'test')
            y_test_dir = os.path.join(dataset_dirpath, 'testannot')
            
            test_dataset = Dataset(
                x_test_dir, 
                y_test_dir, 
                classes = self.classes, 
                #augmentation = get_validation_augmentation(), # only pads to reach 256*256 which is always the case anyway
                preprocessing = get_preprocessing(self.preprocess_input)
            )
        
        # Plot image, label mask and prediction mask
        for i in sample_ids:
    
            image, gt_mask = test_dataset[i]
            image = np.expand_dims(image, axis=0)
            pr_mask = self.model.predict(image)

            print("Test sample " + str(i))
            
            class_labels = np.argmax(gt_mask, axis = 2)
            pr_class_labels = np.argmax(pr_mask, axis = 3)
            
            """
            visualize(image = denormalize(image.squeeze()),
                      labels = denormalize(class_labels.squeeze()),
                      labels_pred = denormalize(pr_class_labels.squeeze()))
            """
            
            visualize_image_and_labels(image = image.squeeze(),
                                       labels = class_labels.squeeze(),
                                       labels_pred = pr_class_labels.squeeze())
    
    def predict_sample(self, sample_id, argmax = False, onehot = False):
        
        image, mask_truth = self.test_dataset[sample_id]
        
        mask_predi = self.model.predict(np.expand_dims(image, axis=0))
        mask_predi = mask_predi.squeeze()
        
        if argmax:
            mask_truth = np.argmax(mask_truth, axis = 2)
            mask_predi = np.argmax(mask_predi.squeeze(), axis = 2)
            
            if onehot:
                mask_truth = np.eye(18)[mask_truth]
                mask_predi = np.eye(18)[mask_predi]
        
        return image, mask_truth, mask_predi
    
    def predict_sample_raw(self, sample_id):

        image, mask = self.test_dataset[sample_id]
        
        mask_pred = self.model.predict(np.expand_dims(image, axis=0))
        
        return image, mask, mask_pred.squeeze()

