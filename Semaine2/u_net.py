import tensorflow as tf
import os
from tqdm import tqdm #tqdm for progress bar
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
#import random

IMG_WIDTH = 572
IMG_HIGH = 572
IMG_CHANNELS = 3

seed= 120
np.random.seed = seed
#train_set = next(os.walk(PATH_TRAIN))[1]

#Encoder
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HIGH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(s)
c1 = tf.keras.layers.DropOut(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c1)
c1 = tf.keras.layers.DropOut(0.1)(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p1)
c2 = tf.keras.layers.DropOut(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c2)
c2 = tf.keras.layers.DropOut(0.1)(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p2)
c3 = tf.keras.layers.DropOut(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c3)
c3 = tf.keras.layers.DropOut(0.2)(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p3)
c4 = tf.keras.layers.DropOut(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c4)
c4 = tf.keras.layers.DropOut(0.2)(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p4)
c5 = tf.keras.layers.DropOut(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c5)

#Decoder
#u6 = tf.keras.layers.Conv2DTranspose((2,2))(c5)

#ModelCheckpoint and others callbacks
checkpointer = tf.keras.callbacks.ModelChekpoint('model_name', verbose=1, save_best_only=True) #save the best model
callbacks = [
        checkpointer,
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs') #define log dir for TensorBoard using
]

#np.squeeze()