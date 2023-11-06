# LL_detection

With the rapid development of society, automobiles have become one of the transportation tools for people. As the number of vehicles are increasing day by day, the number of car accidents are increasing every year. Lane discipline is crucial to road safety for both drivers and pedestrians Following lane lines is one of the most important traffic rules, so detecting them is a significant task while building models for autonomous(self-driving)vehicles.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/db457069-e633-4ee0-8d16-3e5d4846d7ce)

The main objective of this project is to detect lane lines on the roads using the Convolutional Neural Network. In this proposed work, we will detect the lane through which the vehicle is moving. This system is aimed to operate in a real time environment for enhanced safety by faster acquisition and detecting the lanes to assist the driver in controlling and performing scrutinized operations.

The annual increase in vehicle ownerships has caused traffic safety to become an important factor affecting the development of a society. The frequent occurrence of traffic accidents are caused because of human errors .To eliminate all these to a certain extent Smart vehicles (self-driving) were invented.
Lane lines are used to describe the path for self-driving vehicles so, it's necessary to detect lane line on the road surface. Based on the driving lane, determining an effective driving direction for the smart car and providing the accurate position of the vehicle in the lane are possible.

![image](https://github.com/Vijaya0869/LL_detection/assets/109131720/84ffbc7f-02cc-42aa-ac2a-4c2315ae808d)

# Dataset
The dataset is taken from Kaggle
760 Images of lane lines with labels
# Image Pre-Processing
➢ Here we are using Median Filter for the preprocessing. 
➢ Median Filter removes impulse noise from the given image while preserving the image edges. 
➢ Each pixel value is replaced by the median value of the neighboring pixel.
# Image Segmentation 
➢ Image Segmentation is the process of partitioning an image into multiple parts or regions, often based on the characteristics of the pixels in the image. Here we are going to  edge-based segmentation.
# CNN Architecture
In the deep learning techniques, CNN was designed to process multi-dimensional data like images. Here the input data is initially forwarded to a feature extraction network, and then the resultant extracted features are forwarded to the CNN predictor.
# Feature Extraction
Feature Extraction comprises loads of convolutional and pooling layer pairs in CNN. Convolutional layer performs the convolution operation on input data. The pooling layer is used as a dimensionality reduction layer. 
# CNN Predictor
All these extracted features are passed through the neural network .The CNN predictor then works on the basis of the image features and produces the output.


