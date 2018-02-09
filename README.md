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


The image data is originally represented in RGB colour space. A series of colour space transforms were explored to in order to find a space in which the pixels associated with vehicles lie clustered together separated from pixels associated with the background items in the image. This aids the classifier to find an effective decision boundary to separate the classes in this new space. For this purpose image data was first transformed into YCrCb space.

### Histograms of Orientated Gradients (HOG)

The histogram of oriented gradients technique is a popular feature descriptor that uses the gradient information of the pixels to provide some notion of the shape of the object within the image. The [**scikit-image**](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html) package contains an implementation of the HOG technique.

Experiments were performed to find the parameters for the HOG feature extraction technique that maximised the classification accuracy for a common classifier. The following parameters were found to give the best performance.


10 **orientations** : number of orientation bins that the gradient information will be spilt up into the histogram

8 **pixels_per_cell** : Cell size over which each gradient histogram is computed

2 **cells_per_block** : local area over which the histogram counts in a given cell will be normalised

The result of applying the HOG with the above parameters to a singled channelled gray scaled image are shown below.

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/hog_orig.png?raw=true)

![](https://github.com/joshwadd/Vehicle_detection/blob/master/output_images/hog_trans.png?raw=true)


Such a transform results in a returned tensor of size **7x7x2x2x10**. This hog transform is used on each channel of the colour image resulting in a tensor of **3x7x7x2x2x10**. This tensor is the unravelled into a single vector of size **5880** for the HOG feature vector.

### Spatial Information

Spatial information of the image is added to the feature space by taking the original image, resizing it to a smaller resolution (removing teh i

### Colour Histogram







<!--stackedit_data:
eyJoaXN0b3J5IjpbMTcyMzkzMTQyXX0=
-->