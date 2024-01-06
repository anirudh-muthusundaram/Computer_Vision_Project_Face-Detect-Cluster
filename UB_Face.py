'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding Matrix of the faces (may be more than one faces in one image).
            The format of detected bounding Matrix a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.
    # Add your code here. Do not modify the return and input arguments.

    #Matrix to compute face input
    Matrix = face_recognition.face_locations(img)
    detection_results = face_coordinates(Matrix,img)
    return detection_results

def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.
    # Add your code here. Do not modify the return and input arguments.

    #dictionary to append all the required clusters onto the final results
    dict = []
    cluster_results = run_func(dict, cluster_results, imgs, K)    
    return cluster_results



'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)

#traversal function for the grid
def traversal(Matrix,img,above,across_right,below,across_left):
    traversal = []
    for (above, across_right, below, across_left) in Matrix:
        cv2.rectangle(img, (across_left,above), (across_right, below), (0, 255, 0), 2)
        traversal.append([float(across_left),float(above),float(across_right-across_left),float(below-above)])
    return traversal

#face detection variable declaration
def face_coordinates(Matrix,img):
    dict = []
    above = 0
    across_right = 0
    below = 0 
    across_left = 0
    dict = traversal(Matrix,img,above,across_right,below,across_left)
    return dict

#feature extraction for clustering
def feature_extraction(feature, face_clusters):
        return feature[np.random.randint(feature.shape[0], size=face_clusters)]

#distance value extraction for clustering
def value_extraction(value, feature):
        return np.linalg.norm(feature - value, axis=1)

#main function to compute cluster variables
def run_func(dict, cluster_results, imgs, K):
    function_results = [[]]*K
    cluster_val = list(imgs.values())
    cluster_array = np.array(cluster_val)
    position = list(imgs.keys())
    function_results = iterations(dict,cluster_array,position,K)
    return function_results

#iteration function for clustering
def iterations(dict,cluster_array,position,K):
    iteration_results = [[]]*K
    for iterator in cluster_array:
        Matrix = face_recognition.face_locations(iterator)  
        values_end = face_recognition.face_encodings(iterator, Matrix)
        dict.append(values_end[0])
    cluster_array= np.array(dict)
    feature_space = feature_extraction(cluster_array, K)
    clusters = np.zeros(cluster_array.shape[0], dtype=np.float64)
    val = np.zeros([cluster_array.shape[0], K], dtype=np.float64)
    iteration_results = inner_loop(feature_space,clusters,cluster_array,val,position,K)
    return iteration_results

#main loop function that iterates over the cluster results
def inner_loop(feature_space,clusters,cluster_array,val,position,K):
    loop_results = [[]]*K
    number = 1
    while number <= 5000:  
        for pos, location in enumerate(feature_space):
            val[:, pos] = value_extraction(location, cluster_array)
        clusters = np.argmin(val, axis=1)
        for location in range(K):
            feature_space[location] = np.mean(cluster_array[clusters == location], 0)
        number += 1
    for pos in range(K):
        expected_dict = []
        for iter, temp in enumerate(clusters):
            if int(temp) == int(pos):
                expected_dict.append(position[iter])
        loop_results[pos] = expected_dict
    return loop_results
