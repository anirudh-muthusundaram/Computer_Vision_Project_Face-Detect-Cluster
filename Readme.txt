Computer Vision and Image Processing

Face Detection and Cluster Project

This project is comprised of two primary tasks focusing on face detection and clustering, utilizing the capabilities of OpenCV (version 4.5.4) and face recognition libraries.

(Please note that you need to run the project in OpenCV version 4.5.4)

Face Detection:

The first task's objective is to detect faces within a given dataset of images. This is to be achieved using either the OpenCV or face recognition modules. The dataset provided is a subset of the FDDB and includes two folders: a validation folder containing 100 images with ground-truth annotations, and a test folder with an additional 100 images. The core implementation involves writing a function named detect_faces() in the file UB_Face.py. This function should return the bounding boxes of the detected faces in each image. The bounding boxes should be in the format [x, y, width, height].

Face Clustering:

The second task involves the clustering of the cropped faces obtained from Task 1. The objective here is to cluster these faces using a self-implemented clustering algorithm, preferably k-means or a similar approach. The dataset for this task consists of images in the folder named faceCluster_K. The steps to be followed include using face_recognition.face_encodings(img, boxes) to obtain 128-dimensional vectors for each detected face. Importantly, the clustering algorithm must be implemented without the use of specific OpenCV clustering functions or other external libraries that directly implement the clustering algorithm. The number of clusters (K) is indicated in the folder name.