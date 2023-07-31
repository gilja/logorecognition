# logorecognition
Full logo recognition pipeline. Two DNN models are trained: object detection model that finds any logo (or logos) on an image, and logo classification model that is trained to classify between different logo classes.

Steps:

1) Label data

In preparation for the training of object detection model, more than 1000 images were trained containing 27 classes from Flickr27 dataset [1].  
 
In each image, every visible logo was labeled using LabelImg tool [2]. In order to supply high-quality labeled dataset, each bounding box had to be defined as tightly as possible to each logo on every image. Inside the tool, "YOLO" was set as the output format which creates, for each image, a text file consisting of number of rows corresponding to the number of logos within that image. Several columns are created for each logo: label class (number), label name, x and y coordinates of the bounding box normalised to unity and bounding box height and width normalised to unity. 

Only one class was defined (logo), the reason being that object detection algorithm will be trained to detect any logo on any given image. Classifier DNN will be responsible for recognising specific logo classes.  

2) Training the DNN detector

As a bas model for logo detection, YOLOv5 [3] was used where model backbone (10 layers) was frozen. Model head was retrained using recommended setup defined in [3] with 300 epochs and batch size of 64. Stochastic Gradient Descent (SDG) was used as optimiser with momentum 0.937.  


3) Training the DNN classifier

The model used as a base for transfer learning was Xception model [4] which uses weights obtained by training on ImageNet database. Currently, images from Flickr27 dataset are taken and predefined ROIs (regions of interest) which enclose the logo itself are defined. The training was done using TensorFlow through Keras framework. The starting point was the following setup: 

- Cropping images to only include ROIs for training 
- Image augmentation (vertical and horizontal flips, rotations, sheer, zoom, normalization and width and height shift) which produces larger and more general dataset
- Starting learning rate: 1E-4
- batch size: 256
- epochs: 100 
- Image dimensions 224x224 px 
- 80-20 split for training/validation 

Base model: Xception 

Added layers: AveragePooling2D --> Flatten --> Dense (64 neurons using ReLU activation function) --> Dropout (with parameter of. 0.5) --> Dense (with number of neurons equal to the number of added labels, and Softmax as activation function usual for the output layer of NN) 

- Layers of base model frozen  
- Loss function: categorical cross entropy
- optimizer: Adam 

4) Pipeline for combined-classifier model

A python script was written that loads both object detection and classification models discussed in previous steps. A set of previously unseen images was prepared in order to test the performance of the model.  

The first step was feeding a single image through object detection model which returns the bounding boxes of all logos found in that image. For each of the bounding boxes, image crop was made and was fed into the logo classification model in order to obtain the final logo class/classes in an image.  

In order to better understand and evaluate model performance, a green bounding box with a logo class printed within the box (model prediction) was drawn on each image where logo class was recognised by object classification model, while red bounding box without logo class printed was drawn if logo class was not recognised by the object classification model. 

The final result outperforms the single-step models which rely solely on classifying logos from the training dataset.

[1] https://www.kaggle.com/datasets/sushovansaha9/flickr-logos-27-dataset
[2] https://github.com/HumanSignal/labelImg
[3] https://github.com/ultralytics/yolov5
[4] https://arxiv.org/abs/1610.02357