# Classic Vehicle Detection

## Project Overview

This project builds a vehicle detection pipeline using classic techniques from computer vision and machine learning. Such techniques have fell out of favour in recent times and instead deep learning systems trained end to end have shown to produce better performance in terms of accuracy and computational efficiency of implementation. The point of this project then (aside from nostalgia) is to gain experience with the feature extraction and feature engineering process that underpinned the performance of object detection techniques of old. Doing so also gives an appreciation and intuition of the learnt feature extracting layers found in deep learning architectures.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/feature_extractor.py`      | `CameraCalibration` class used to take an input image and extract the features from a given window region in the image. |
| `Code/vehicle_detection.py`     | `VehicleDetection` class used to apply a SVM classifier across an image through windows of various sizes to detect the presence of all vehicles. |
| `Code/vehicle_detection.ipynb`   | Ipython notebook used to train a linear SVM on a training data set of vehicle and non vehicle images|

## Data Set

The key component to the object detection pipeline is a robust image classifier, capable of detecting if a given image contains a vehicle or not. To do this I train a binary classifier on a data set made up of 17760 rgb images 64x64 pixels in size. In this dataset 8792 are images that contain a vehicle and 8968 are images that contain non-vehicle images taken from a front facing camera on a road vehicle.

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/data_set.jpg?raw=true)



## Feature Extraction

Deciding on the best features to extract from the image for robust classification was done with a combination of trail, error and intuition. The final set of features extracted I decided upon were made up of a combination of **HOG (Histogram of Oriented Gradients)**,  **spatial information** and **histograms of colour channels**.  All implementations for the feature extractions are found in the `FeatureExtraction` class. This class takes an input image on initialisation and extracts the desired features for any desired region of the image.

### Colour Space Exploration

The image data is orginally represented in RGB 

### Histograms of Orientated Gradients (HOG)

The histogram of oriented gradients technique is a popular feature descriptor







<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk4NzcyNjgzMF19
-->