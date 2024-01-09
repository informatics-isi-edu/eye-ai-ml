import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd

def load_model(model_path):
    IMG_SIZE = (300, 300)

    preprocess_input = tf.keras.applications.vgg19.preprocess_input

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                include_top=False,

                                                weights='imagenet')

    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 17

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    intermediate_layer_1 = tf.keras.layers.Dense(128, activation='relu')
    intermediate_layer_2 = tf.keras.layers.Dense(128, activation='relu')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)#, activation='sigmoid')

    inputs = tf.keras.Input(shape=(300, 300,  3))
    # x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = intermediate_layer_1(x, )#activation='relu')
    x = tf.keras.layers.Dropout(0.2)(x)
    x = intermediate_layer_2(x)#, activation='relu')
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    from keras.callbacks import Callback, EarlyStopping

    val_acc_es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=3, restore_best_weights=True)
    # train_auc_es = EarlyStopping(monitor=auc, mode='max', verbose=1, patience=6, restore_best_weights=True)
    # val_auc_es = EarlyStopping(monitor='val_auc', mode='max', verbose=1, patience=4, restore_best_weights=True)

    SAVED_MODEL_NAME_acc_vgg19_300_auc = 'Merged Cropped Porper or Not Dataset TLBR 95 VGG19 Val Accuracy.hdf5'

    model_checkpoint_callback_val_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVED_MODEL_NAME_acc_vgg19_300_auc,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    auc  = tf.keras.metrics.AUC()

    check = tf.keras.callbacks.ModelCheckpoint('AUC '+SAVED_MODEL_NAME_acc_vgg19_300_auc,
                                            monitor='val_auc',  # even use the generated handle for monitoring the training AUC
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='max')  # determine better models according to "max" AUC.


    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy', 'AUC'])

    # Load the pre-trained model

    model.load_weights(model_path)
    return model


