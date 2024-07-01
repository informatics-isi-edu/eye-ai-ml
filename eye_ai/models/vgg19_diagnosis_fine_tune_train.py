import sys
import argparse
import numpy as np
import random
import os
import gc
from pathlib import Path, PurePath
import logging
import json

import pandas as pd
from sklearn.utils import class_weight

import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ... (keep all the existing imports and helper functions)

def fine_tune_model(train_path, valid_path, test_path, output_path, best_params, model_name, original_model_path):
    set_seeds()

    train_generator, validation_generator, test_generator = get_data_generators(train_path, valid_path, test_path, best_params)
    
    # Load the previously trained model
    model = load_model(original_model_path, custom_objects={'f1_score_normal': f1_score_normal})
    
    # Unfreeze all layers for fine-tuning
    for layer in model.layers:
        layer.trainable = True
    
    # Compile model with a lower learning rate
    fine_tuning_lr = best_params['fine_tuning_learning_rate_adam'] * 0.1  # Reduce learning rate
    optimizer = Adam(learning_rate=fine_tuning_lr)
    model.compile(
        optimizer=optimizer,
        loss=BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc_score"),
            f1_score_normal,
            tf.keras.metrics.BinaryAccuracy(name="accuracy_score"),
        ]
    )
    
    # Training
    class_weights = None
    if best_params['use_class_weights']:
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
        class_weights = dict(enumerate(class_weights))

    num_workers = os.cpu_count()

    training_log = model.fit(
        train_generator,
        epochs=50,  # Reduce number of epochs for fine-tuning
        validation_data=validation_generator,
        batch_size=best_params['batch_size'],
        class_weight=class_weights,
        workers=num_workers,
        use_multiprocessing=True,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            EarlyStopping(monitor='val_roc_auc_score', mode='max', verbose=1, patience=5, restore_best_weights=True),
        ],
    )

    # Evaluate the model on the test set
    results = model.evaluate(test_generator)
    logging.info(f"Test results - {results}")
    print(f"Model Eval results: {results}")

    # Save the fine-tuned model
    fine_tuned_model_name = f'{model_name}_fine_tuned'
    model.save(os.path.join(output_path, f'{fine_tuned_model_name}.h5'))
    
    # Save training history
    hist_df = pd.DataFrame(training_log.history) 
    hist_df.to_csv(os.path.join(output_path, f'training_history_{fine_tuned_model_name}.csv'), index=False)
    
    logging.info(f"{fine_tuned_model_name} Model fine-tuned, Model and training history are saved successfully.")

def main(train_path, valid_path, test_path, output_path, best_hyperparameters_json_path, model_name, original_model_path):
    logging.basicConfig(level=logging.INFO)
    
    # Load best parameters from JSON
    with open(best_hyperparameters_json_path, 'r') as file:
        best_params = json.load(file)

    fine_tune_model(train_path, valid_path, test_path, output_path, best_params, model_name, original_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training images')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the validation images')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test images')
    parser.add_argument('--output_path', type=str, required=True, help='Path where the fine-tuned model should be saved')
    parser.add_argument('--best_hyperparameters_json_path', type=str, required=True, help='Path to the JSON file with best hyperparameters')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the fine-tuned model')
    parser.add_argument('--original_model_path', type=str, required=True, help='Path to the original trained model')
    args = parser.parse_args()

    main(args.train_path, args.valid_path, args.test_path, args.output_path, args.best_hyperparameters_json_path, args.model_name, args.original_model_path)
