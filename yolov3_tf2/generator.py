import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import yolov3_tf2.autoaugment_utils as autoaugment

import cv2
from PIL import Image

""" Generator Class for Yolo from a CSV (like RetinaNet)

Making this so it is easier to run Yolov3 and so we can include augmentation
"""
class YoloCSVGenerator(keras.utils.Sequence):
    def __init__(
        self, 
        csv_data_file, 
        csv_class_file, 
        anchors,
        anchor_masks,
        img_size=416, 
        shuffle=False,
        batch_size=8,
        yolo_max_boxes=100,
        augment=False
    ):
        self.dim            = (img_size, img_size, 3)
        self.shuffle        = shuffle
        self.batch_size     = batch_size
        self.yolo_max_boxes = yolo_max_boxes
        self.anchors        = anchors
        self.anchor_masks   = anchor_masks
        self.augment        = augment

        # Load the classes we care about
        self.class_map = pd.read_csv(csv_class_file, names=['class', 'value'])
        self.class_dict = {}
        for i in range(self.class_map.shape[0]):
            self.class_dict[self.class_map['class'][i]] = self.class_map['value'][i]

        # Get complete dataset
        self.full_data = pd.read_csv(csv_data_file, names=['file', 'x1', 'y1', 'x2', 'y2', 'class'])
        self.image_names = []
        for i in self.full_data['file']:
            if i not in self.image_names:
                self.image_names.append(i)

        # Now that we have the set of images, filter out for only the objects we care about
        self.full_data = self.full_data[self.full_data['class'].isin(self.class_dict)]

        self.on_epoch_end()

    """Number of batches per epoch"""
    def __len__(self):
        return int(np.floor(len(self.image_names) / self.batch_size))

    """Updates indices after each epoch"""
    def on_epoch_end(self):
        self.indices = np.arange(len(self.image_names))

        if self.shuffle:
            np.random.shuffle(self.indices)

    """Generate one batch of data"""
    def __getitem__(self, index, debug=False):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        imgs_using = [self.image_names[k] for k in indices]

        X, y = self.__data_generation(imgs_using, debug)

        return X, y

    """Geneates data containing batch_size samples"""
    def __data_generation(self, imgs_using, debug=False):
        X = np.empty((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size, self.yolo_max_boxes, 5))

        for i, img in enumerate(imgs_using):
            # Read in and resize the image
            img_array = cv2.cvtColor(cv2.imread(img), cv2.COLOR_RGB2BGR)

            y_tent = self.__read_annotations(img, img_array.shape)
            
            # Resize after processing the annotations, because we need the original image size
            # in order to normalize properly
            img_array = cv2.resize(img_array, self.dim[:2])

            # We should apply any data augmentation right here
            # Default to using policy v0 for now
            if self.augment:
                # Wrap this in a try catch because sometimes it gives an error
                try:
                    img_array, y_tent = apply_augmentation(
                        x = img_array, 
                        y = y_tent,
                        augmentation_name='v0'
                    )
                except:
                    print('There was an error with augmentation')

            if debug:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Rectangle

                fig,ax = plt.subplots(1)
                ax.imshow(img_array/255.0)

                for ii in y_tent:
                    coords = [i * img_array.shape[0] for i in ii]
                    rect = Rectangle(
                        (coords[0], coords[1]), 
                        coords[2] - coords[0], 
                        coords[3] - coords[1], 
                        linewidth=1,edgecolor='b',facecolor='none')
                    ax.add_patch(rect) 

                plt.show()

            X[i,] = img_array / 255.0
            y[i,:y_tent.shape[0],:y_tent.shape[1]] = y_tent

        # Now transform the targets
        y_out = transform_targets(y, self.anchors, self.anchor_masks, self.dim[0])

        return X, y_out

    def __read_annotations(self, img, img_size):
        annotations_for_image = self.full_data[self.full_data['file']==img].copy()
        annotations_for_image['class'] = [self.class_dict[k] for k in annotations_for_image['class']]
        
        annotations_as_np = np.array(annotations_for_image[['x1', 'y1', 'x2', 'y2', 'class']]).astype(np.float32)
        annotations_as_np[:,1] = annotations_as_np[:, 1]/img_size[0]
        annotations_as_np[:,3] = annotations_as_np[:, 3]/img_size[0]
        annotations_as_np[:,0] = annotations_as_np[:, 0]/img_size[1]
        annotations_as_np[:,2] = annotations_as_np[:, 2]/img_size[1]
        
        return annotations_as_np

"""Taken directly from yolo dataset.py"""
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = np.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = np.zeros(
        (N, grid_size, grid_size, np.shape(anchor_idxs)[0], 6))

    anchor_idxs = anchor_idxs.astype(np.int32)

    for i in np.arange(N):
        for j in np.arange(np.shape(y_true)[1]):
            # If x2 is equal to 0, then this is a blank
            if np.equal(y_true[i][j][2], 0):
                continue

            anchor_eq = np.equal(
                anchor_idxs, y_true[i][j][5].astype(np.int32))

            if np.any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = np.where(anchor_eq)
                grid_xy = (box_xy // (1/grid_size)).astype(np.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                y_true_out[i,grid_xy[1], grid_xy[0], anchor_idx[0][0],:] = [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]]

    return y_true_out

def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = anchors.astype(np.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = np.tile(np.expand_dims(box_wh, -2),
                     (1, 1, np.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = np.minimum(box_wh[..., 0], anchors[..., 0]) * \
        np.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = np.argmax(iou, axis=-1).astype(np.float32)
    anchor_idx = np.expand_dims(anchor_idx, axis=-1)

    y_train = np.concatenate([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)

"""Apply the Google Brain augmentation policies they defined"""
def apply_augmentation(x, y, augmentation_name='test'):
    image = tf.cast(x, dtype=tf.uint8)
    bbox = y[:,:4]
        
    # For some reason they flip the min and max coords **facepalm**
    # Order is xmin, ymin, xmax, ymax. So we need to flip them (and then we will flip back smh)    
    bbox_in = tf.concat([bbox[:,1], bbox[:,0], bbox[:,3], bbox[:,2]], -1)
    bbox_in = tf.reshape(bbox_in, (4, -1))
    bbox_in = tf.transpose(bbox_in)
    
    # Apply augmentation
    res = autoaugment.distort_image_with_autoaugment(
        image = image,
        bboxes = bbox_in,
        augmentation_name=augmentation_name
    )
    
    # Convert the bbox back to coordinates it should be
    bbox_out = tf.concat([res[1][:,1], res[1][:,0], res[1][:,3], res[1][:,2]], -1)
    bbox_out = tf.reshape(bbox_out, (4, -1))
    bbox_out = tf.transpose(bbox_out)
    
    # We need to get rid of the transormations on bounding boxes that are all 0s.
    # This resets them to all 0s
    bbox_out = tf.math.multiply(bbox_out, tf.math.ceil(bbox))
    
    # We need to add the labels column back to the bounding box output
    final_out = tf.concat([bbox_out, tf.reshape(y[:,4], (-1, 1))], axis=1)
    return tf.cast(res[0], dtype=tf.float32), final_out

