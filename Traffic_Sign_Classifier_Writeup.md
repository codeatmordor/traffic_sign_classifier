# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./German_Traffic_Signs/GerTrafSign1.jpeg "Sign 1"
[image2]: ./German_Traffic_Signs/GerTrafSign2.jpeg "Sign 2"
[image3]: ./German_Traffic_Signs/GerTrafSign3.jpeg "Sign 3"
[image4]: ./German_Traffic_Signs/GerTrafSign4.jpeg "Sign 4"
[image5]: ./German_Traffic_Signs/GerTrafSign5.jpeg "Sign 5"
[image6]: ./German_Traffic_Signs/GerTrafSign6.jpeg "Sign 6"
[image7]: ./German_Traffic_Signs/GerTrafSign7.jpeg "Sign 7"
[image8]: ./Vis.jpeg "Visualization"
[image9]: ./German_Traffic_Signs/1.png "1"
[image10]: ./German_Traffic_Signs/2.png "2"
[image11]: ./German_Traffic_Signs/3.png "3"
[image12]: ./German_Traffic_Signs/4.png "4"
[image13]: ./German_Traffic_Signs/5.png "5"
[image14]: ./German_Traffic_Signs/6.png "6"
[image15]: ./German_Traffic_Signs/7.png "7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Numpy library is used to enumerate summary of dataset.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

All the images were plotted in 5*8 setup.

![alt text][image8]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided shuffle my training dataset and then normalization was used as a preprocessing step. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Layer 1     	| Output 28x28x6 	|
| ReLU					|					Activate Layer 1							|
| Max pooling	      	| 28*28*6 to 14*14*6 				|
| Layer 2	    | Output 10*10*6      									|
| ReLU		| Activate Layer 2        									|
| Max Pooling				| 10*10*6 to 5*5*6        									|
|	Flatten					|5*5*6 , Output = 400												|
|	Connected Layer 3					|400 to 120												|
|	ReLU					|Activate Layer 3											|
|	Connected Layer 4					|120 to 84												|
|	ReLU					|Activate Layer 4												|
|	Connected Layet 5					|84 to 43												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Model traning parameters:
Epoch - 20
Batch Size - 128
Learning Rate - 0.001
Sigma - 0.1
I have used same LeNet model explained during lectures, this model has 2 CNNs and 3 connected layers.Input for this multilayer model is image of size (32,32,3) and output is 43 which disctinct labels.ReLU is used on output of each layer for activation.Flatten is used to convert of output of 2D CNN. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of .9816

This model has accurately predicted output for new german traffic sign images with 0.5714 probability.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are Seven German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image2] ![alt text][image1]
![alt text][image3]

German Traffic Signs after Resizing to (32,32,3)

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image14] ![alt text][image13] ![alt text][image12]
![alt text][image15]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|Pedestrian     		| Speed limit (30km/h)   									| 
| No entry     			| No entry 										|
| No Pass					| Go straight or left											|
| Turn right ahead	      		| Turn right ahead					 				|
| Yield			| Yield      							|
| Road work			| Road work      							|
| Stop			| Stop      							|


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


