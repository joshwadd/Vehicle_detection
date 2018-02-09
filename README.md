# Classic Vehicle Detection

## Project Overview

This project builds a vehicle detection pipeline using classic techniques from computer vision and machine learning. Such techniques have fell out of favour in recent times and instead deep learning systems trained end to end have shown to produce better performance in terms of accuracy and computational efficiency of implementation. The point of this project then (aside from nostalgia) is to gain experience with the feature extraction and feature engineering process that underpinned the performance of object detection techniques of old. Doing so also gives an appreciation and intuition of the learnt feature extracting layers found in deep learning architectures.

| File                                | Description                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `Code/feature_extractor.py`      | `CameraCalibration` class used to remove any distortions in the image associated with the imaging hardware. |
| `Code/edge_detection.py`     | Set of gradient and colour space transformation routines used to detect pixels associated with lane lines. |
| `Code/perspective.py`   | `PerspectiveTransformation` class to transform the perspective of the front facing camera image to a overhead lane view. |
| `Code/line.py` | `Line` class to represent a single lane line |dete
| `Code/window.py`        | `Window` class to represent a single scanning window used to detect pixels likely associated with lane lines.|
| `Code/lane_detection.py`      | `LaneDectection` class implementing a lane detection processing pipeline to a consecutive series of frames from a video. |
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgzOTY4MTQ1Nl19
-->