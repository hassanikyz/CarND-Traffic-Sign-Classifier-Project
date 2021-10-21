# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization2.jpg "Visualization 2"
[image2]: ./examples/grayscale1.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/test_image_from_web.jpg "Test Images From Web"
[image5]: ./examples/random_samples.jpg "Random Samples From Training Data"
[image6]: ./examples/top-5-prediction.jpg "Top 5 Predictions"
[image7]: ./examples/visualization1.jpg "Visualization 1"
[image8]: ./examples/mode-tain1.jpg "Model Training Start"
[image9]: ./examples/mode-tain2.jpg "Model Training End"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and pandas libraries to collect summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape (shape of traffic sign image) = (32, 32, 3)
Number of categories/classes = 43
     ClassId                                           SignName
        0                               Speed limit (20km/h)
        1                               Speed limit (30km/h)
        2                               Speed limit (50km/h)
        3                               Speed limit (60km/h)
        4                               Speed limit (70km/h)
        5                               Speed limit (80km/h)
        6                        End of speed limit (80km/h)
        7                              Speed limit (100km/h)
        8                              Speed limit (120km/h)         
        9                                         No passing
       10       No passing for vehicles over 3.5 metric tons
       11              Right-of-way at the next intersection
       12                                      Priority road
       13                                              Yield
       14                                               Stop
       15                                        No vehicles
       16           Vehicles over 3.5 metric tons prohibited
       17                                           No entry
       18                                    General caution
       19                        Dangerous curve to the left
       20                       Dangerous curve to the right
       21                                       Double curve
       22                                         Bumpy road
       23                                      Slippery road
       24                          Road narrows on the right
       25                                          Road work
       26                                    Traffic signals
       27                                        Pedestrians
       28                                  Children crossing
       29                                  Bicycles crossing
       30                                 Beware of ice/snow
       31                              Wild animals crossing
       32                End of all speed and passing limits
       33                                   Turn right ahead
       34                                    Turn left ahead
       35                                         Ahead only
       36                               Go straight or right
       37                                Go straight or left
       38                                         Keep right
       39                                          Keep left
       40                               Roundabout mandatory
       41                                  End of no passing
       42  End of no passing by vehicles over 3.5 metric ...

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram and bar charts showing how the data is labeled across 43 different classes


![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it speeds up analysis and is known to retain the kind of features that convolutional neural network is suited to find. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Initial model was based on LENET however, its accuracy at most went up to 88 or so after playing with hyperparameters like batch size, training rate. Adding droputs increased accuracy to around 91 but it finally crossed 93 when preprocessing of images was added -- mainly grayscaling and normalization. 

My final model consisted of the following layers:

| Layer         						|     Description	        					| 
|:---------------------:				|:---------------------------------------------:| 
| Raw Input         					| 32x32x3 RGB image   							| 
| Pre-process(Grayscale and normalized) | 32x32x1 RGB image   							| 
| Convolution 3x3     					| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU									|												|
| Max pooling	      					| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    				| etc.      									|
| RELU									|												|
| Max pooling	      					| 2x2 stride,  outputs   5x5x16 				|
| Flatten and Dropout	        		| 2x2 stride,  outputs   400     				|
| Fully connected						| 400 and Output 120   							|
| RELU and dropout						|												|
| Fully connected						| 120 and Output 84   							|
| RELU and dropout						|												|
| Fully connected						| 84 and Output 43   							|
|										|												|
|										|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Initially I simply used Lenet architecture without any preprocessing where with adjustments were required in handling RGB images.
That didn't give high enough accuracy, it was somewhere in the high 80s range. By reducing the learning rate from 0.001 to 0.0009 I saw accuracy improved a bit. Initially I used batchsize of 128 and played with various values of epocs e.g 50, 60, 70. However, those didn't increase accuracy significantly. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
*Train Accuracy = 0.987
*Test Accuracy = 0.937
*Valid Accuracy = 0.963

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LENET-5 was given and suggested by instructors and it was already setup for MNIST data. There are many parallels between MNIST and Traffic Sign classification problem. Using that as the starting point therefore made sense.

* What were some problems with the initial architecture?
It didn't have dropouts which proved to be a bit more effective. As-is version of LENET-5 didn't get accuracy above 0.88 in my runs. Additional convolution layer also helped taking the accuracy up to 96%

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Mainly dropouts after relu and an additional convolutional layer was added to improve accuracy and reduce over-fitting.

* Which parameters were tuned? How were they adjusted and why?
Hyperparameters impact wasn't that big. Tried few learing rates and batch sizes. The biggest difference came from architectural updates. Afer trying 0.0006, 0.0009 and I settled for 0.001 for learning rate and batch size of 28. Dropout was set at 0.5

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
It reduce chances of overfitting and thus enable model to scale better.

If a well known architecture was chosen:
* What architecture was chosen?
Lenet-5 as this was given by Udacity to be a good starting point. It has performed well for MNIST (hardwritten digits recognition).

* Why did you believe it would be relevant to the traffic sign application?
Several aspects of LENET-5's original design intent as similar to traffic signs. The traffic signs are classified into a discrete set of 43 (MNIST is 10). Signs have relatively stable feature pattern just like digits.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
There isn't a huge difference between training and validatoin accuracy which shows mode isn't overly fit by the training data.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set. However, after I added modified images (rotation and perspective transformed), that number went up to 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


