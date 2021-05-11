import numpy as np
import sys
import os
import csv   
from config import MODEL_PATH, STANDARDPATH
import pickle

import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import Sequential

def build_model(ml_options, X_train,X_test=None, y_train=None,y_test=None):

    feature_nr = X_train.shape[1] 

    deep_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(feature_nr,)), # die Input shape hängt von der Anzahl an features ab
        tf.keras.layers.Dense(3, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    
    return deep_model

def fit_model(X_train, y_train, X_test, y_test,deep_model, ml_options): 

    train_x = np.asarray(X_train).astype('float32')
    train_y = np.asarray(y_train).astype('float32')
    validation_x = np.asarray(X_test).astype('float32')
    validation_y = np.asarray(y_test).astype('float32')

    deep_model.compile(optimizer=ml_options["optimizer"],
              loss=ml_options["loss"],
              metrics=ml_options["metrics"])
    
    history = deep_model.fit(train_x, train_y, 
                         epochs=20, 
                         verbose = 1, 
                         batch_size=20, 
                         validation_data=(validation_x, validation_y))
    
    return deep_model

def predict(X_test, y_test, deep_model, ml_options):

    validation_x = np.asarray(X_test).astype('float32')
    validation_y = np.asarray(y_test).astype('float32')

    y_pred = np.asarray([i for i in deep_model.predict_classes(validation_x)[:,0]]).astype('float32')
    y_prediction = np.zeros((len(y_test), 3))
    y_prediction[:,0] = y_pred
    y_prediction[:,1] = validation_y[:]

    counter_class1_correct = 0
    counter_class0_correct = 0
    counter_class1_incorrect = 0
    counter_class0_incorrect = 0

    for i in range(len(validation_y)):
        if y_prediction[i,0] == y_prediction[i,1]:
            y_prediction[i,2] = 1
            if y_prediction[i,1] == 1:
                counter_class1_correct += 1
            else:
                counter_class0_correct += 1
        else:
            y_prediction[i,2] = 0
            if y_prediction[i,1] == 1:
                counter_class1_incorrect += 1
            else:
                counter_class0_incorrect += 1
    print(counter_class1_correct, counter_class0_correct, counter_class1_incorrect,counter_class0_incorrect)
    """ Calculate accuracy scores """

    accuracy = y_prediction.mean(axis=0)[2]
    accuracy_class1 = counter_class1_correct / (counter_class1_correct + counter_class1_incorrect) # Recall
    accuracy_class0 = counter_class0_correct / (counter_class0_correct + counter_class0_incorrect)
    balanced_accuracy = (accuracy_class1 + accuracy_class0) / 2
    try:
        precision = counter_class1_correct / (counter_class1_correct + counter_class0_incorrect)
    except ZeroDivisionError:
        precision = 0.0
    try: 
        f1_score = 2 * ((accuracy_class1 * precision)/(accuracy_class1+precision))
    except ZeroDivisionError:
        f1_score = 0.0
    log_loss_value = deep_model.evaluate(validation_x, validation_y)[0]

    print('Round Number: ', str(ml_options["seed"]), '\nAccuracy: ', str(accuracy), '\nAccuracy_class0: ', str(accuracy_class1), '\nAccuracy_class1/Recall: ', 
        str(accuracy_class0), '\nPrecision: ', str(precision),'\nF1 Score: ', str(f1_score),'\nBalanced Accuracy: ', str(balanced_accuracy), '\nLog Loss value: ', str(log_loss_value))

    savepath = os.path.join(STANDARDPATH, 'outcomes_NN.csv')
    if not os.path.exists(savepath):
        header = ['model', 'seed/run', 'accuracy', 'accuracy_class1/recall', 'accuracy_class0', 'precision', 'f1_score','balanced_accuracy', 'log_loss_value']
        with open(savepath, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write outcome rows
            writer.writerow([ml_options["model_name"],ml_options["seed"], accuracy, accuracy_class1, accuracy_class0, precision, f1_score,balanced_accuracy, log_loss_value])
    else:
        with open(savepath, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ml_options["model_name"],ml_options["seed"], accuracy, accuracy_class1, accuracy_class0, precision, f1_score, balanced_accuracy, log_loss_value])
     
     
    outcome_list = [accuracy, accuracy_class1, accuracy_class0, precision, f1_score, balanced_accuracy,log_loss_value]

    return outcome_list
