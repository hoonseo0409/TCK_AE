import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D, SimpleRNN, GRU, LSTM, Embedding
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

def get_input_image(path, start, end, width, height):
    """get 4 input image from dataset (1 original, 3 next frames) 
    
    Arguments:
        path {[String]} -- [path of dataset]
        start {[int]} -- [start index]
        end {[int]} -- [end index]
        width {[int]} -- [width of image after resize]
        height {[int]} -- [height of image after resize]
        x_container {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    result = []
    # should be [# timepoints T, # samples, # features V]
    for i in range(start,end):
        # x_container.append(cv2.resize(cv2.imread(path + str(i).zfill(6) + '.png'),(width, height)))
        # x_container.append(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_01.png'),(width, height)))
        # x_container.append(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_02.png'),(width, height)))
        # x_container.append(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_03.png'),(width, height))) 
        sample_list = []
        sample_list.append(cv2.cvtColor(cv2.resize(cv2.imread(path + str(i).zfill(6) + '.png'),(width, height)),cv2.COLOR_BGR2GRAY).flatten())
        sample_list.append(cv2.cvtColor(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_01.png'),(width, height)),cv2.COLOR_BGR2GRAY).flatten())
        sample_list.append(cv2.cvtColor(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_02.png'),(width, height)),cv2.COLOR_BGR2GRAY).flatten())
        sample_list.append(cv2.cvtColor(cv2.resize(cv2.imread(path + str(i).zfill(6) + '_03.png'),(width, height)),cv2.COLOR_BGR2GRAY).flatten())    
        result.append(np.stack(sample_list))
    result = np.transpose(np.stack(result), axes = [1, 0, 2])
    return result

def get_input_image_one(path, start, end, width, height, x_container):
    """[get 1 input image from dataset (1 original)] 
    
    Arguments:
        path {[String]} -- [path of dataset]
        start {[int]} -- [start index]
        end {[int]} -- [end index]
        width {[int]} -- [width of image after resize]
        height {[int]} -- [height of image after resize]
        x_container {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    for i in range(start,end):
        x_container.append(cv2.resize(cv2.imread(path + str(i).zfill(6) + '.png'),(width, height)))      
        
    x_container = np.stack(x_container)
    return x_container

def get_label_fromfile(path, start, end):
    """[get label of dataset from file]
    
    Arguments:
        path {[String]} -- [path of label file]
        start {[int]} -- [start index]
        end {[int]} -- [end index]
    
    Returns:
        [type] -- [description]
    """

    label = []

    for i in range(start,end):
        f = open(path + str(i).zfill(6) + ".txt","r")
        found = False
        for line in f:
            fields = line.split(" ")
            if fields[0] == "Car":
                #label.append([i, fields[11], fields[12], fields[13]])
                label.append([float(fields[11]), float(fields[12]), float(fields[13])])
                found = True
                break
        f.close()
        if found == False:
            #label.append([i, '0.00', '0.00', '0.00'])
            label.append([0.00, 0.00, 0.00])
            
    return np.array(label)

def get_boundarybox_fromfile(path, start, end):
    """[get boundary box of object from file]
    
    Arguments:
        path {[String]} -- [path of label file]
        start {[int]} -- [start index]
        end {[int]} -- [end index]
    
    Returns:
        [int, int, int, int] -- [2D bounding box of object in the image (0-based index):contains left, top, right, bottom pixel coordinates]
    """

    box = []

    for i in range(start,end):
        f = open(path + str(i).zfill(6) + ".txt","r")
        found = False
        for line in f:
            fields = line.split(" ")
            if fields[0] == "Car":
                #box.append([i, fields[4], fields[5], fields[6], fields[7]])
                box.append([float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])])
                found = True
                break
        f.close()
        if found == False:
            box.append([0.00, 0.00, 0.00, 0.00])
            
    return box   

def concatVector(start, end, data1, data2):

    out = []

    for i in range(end - start):
        out.append(np.concatenate((data1[i], data2[i])))

    return np.array(out)

def concatVectorBox(start, end, data1, data2, data3):

    out = []

    for i in range(end - start):
        out.append(np.concatenate((data1[i], data2[i], data3[i])))

    return np.array(out)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(np.uint8(y_pred) - np.uint8(y_true)))) 

