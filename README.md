# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/sign.jpg "Traffic sign"
[image2]: ./images/training_dist.jpg "Training sample distribution"
[image3]: ./images/validation_dist.jpg "Validation sample distribution"
[image4]: ./images/test_dist.jpg "Test sample distribution"
[image5]: ./images/sign_gray.jpg "Grayscale Traffic Sign"
[image6]: ./images/sign_gray_100.jpg "100 Training Samples of 10 Class"
[image7]: ./images/shift.jpg "Random shift"
[image8]: ./images/rotate.jpg "Random rotate"
[image9]: ./images/warp.jpg "Random warp"
[image10]: ./images/blur.jpg "Random blur"
[image11]: ./images/noise.jpg "Random noise"
[image12]: ./images/training_aug_dist.jpg "Augumented training sample distribution"
[image13]: ./images/train_valid_acc.jpg "Training Validation accuracy"
[image14]: ./images/28.jpg "Children crossing"
[image15]: ./images/17.jpg "No entry"
[image16]: ./images/13.jpg "Yield"
[image17]: ./images/14.jpg "Stop"
[image18]: ./images/12.jpg "Priority road"
[image19]: images/sign_pred.jpg "Sign prediction"
[image20]: images/conv1.jpg "conv1"
[image21]: images/conv1_act.jpg "conv1_act"
[image22]: images/conv1_pool.jpg "conv1_pool"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle and csv library to load pickle and csv file, and used basic list function to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **(32, 32, 3)**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

![alt text][image1]

Here are the training, validation and test sample distribution.

| Sample distribution   |
|:---------------------:|
| ![alt text][image2]   |
| ![alt text][image3]   |
| ![alt text][image4]   |

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because we can reduce the color channel to one to make things more easier.

Here is an example of a traffic sign image after grayscaling.

![alt text][image5]

Here is another view of 100 training samples within 10 classes.

![alt text][image6]

As a last step, I normalized the image data to [0..1] because it can make the following CNN calculation more smooth.

I decided to generate additional data because I found that some class in training data are some small. 

To add more data to the the data set, I used the following techniques.

Here are some augmented image example.

| Method         		|     Result	       | 
|:---------------------:|:--------------------:| 
| Random shifted        | ![alt text][image7]  |
| Random rotated       | ![alt text][image8]   |
| Random warped         | ![alt text][image9]  |
| Random blurred        | ![alt text][image10] |
| Random noised         | ![alt text][image11] |

Here is the augmented training sample distribution.

![alt text][image12]

The difference between the original data set and the augmented data set is that I add some random shifted, rotated, warped, blurred and noised samples to the training sample to let small class as close as average.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5       | 5x5 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 5x5 stride, valid padding, outputs 10x10x16   |
| RELU					|	                                            |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 3x3	    | 3x3 stride, valid padding, outputs 3x3x64     |
| RELU					|												|
| Max pooling	      	| 2x2 ksize, 1x1 stride,  outputs 2x2x64        |
| Flatten		        | outputs 256        							|
| Fully connected		| outputs 120      								|
| RELU					| 									            |
| Dropout				| 								                |
| L2 regularizer        |                                               |
| Fully connected		| outputs 84      								|
| RELU					| 									            |
| Dropout				| 								                |
| L2 regularizer        |                                               |
| Fully connected		| outputs 43      								|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My model, modified from LaNet, contains 3 convolution layer and 3 fully connected layer. Lost function is cross entropy. Used the Adam as optimizer with following hyperparameters.
```
EPOCH = 50
BATCH_SIZE = 128
droupout = 0.5
rate = 0.001
beta = 0.001 (l2 regularizer) 
```

Besides, since there are some randomization during the training. I trained 10 times and pick the best one as final model.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **99.7%**
* validation set accuracy of **97.8%**
* test set accuracy of **94.4%**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    - I used LaNet as my baseline CNN architecture, because in the LaNet lab I found its validation accuracy is 99% in MNIST dataset.
* What were some problems with the initial architecture?
    - In LaNet CNN, its first layer input shape is (32x32x1), which means data are grayscale. If we want to use its architecture, we first need to convert our data to gray.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    - To improve the performance, I first add 3rd convolution layer with RELU activation and max pooling to detect more detail things. Then added dropout to every fully connected layer to drop tiny neuron. Finally, in order to prevent overfitting, I added L2 regularizer to every fully connected layer.
* Which parameters were tuned? How were they adjusted and why?
    - I tried to add more convolution layer, change ksize and stride. Tried Adagram optimizer. Tried different normalization method. Added dropout to different layer. Also tried to modify the rate and EPOCH, found that the weight has some initial randomization problem, so run 10 rounds and pick the best model as my final choice. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    - I found convolution layer is good at image detection. In my experiment, the performance of deeper convolution layer is better than fully connected layer. But more deeper convolution layer means more training time. It may be relevant to the network architecture. More deeper convolution will detect more detail portion. Besides, I also found that dropout can really improve the performance, since it eliminate many small useless neurons to affect final prediction.

If a well known architecture was chosen:
* What architecture was chosen?
    - I think for image detection, CNN would be a good one. It first use different convolution layer to detect different small object, then combine them to detect large one. Finally use fully connected layer to combine these convolution layer to output the final prediction.
* Why did you believe it would be relevant to the traffic sign application?
    - CNN is good in image detection application. It can detect any kind of target object. Since our training sample are all traffic sign, so the model will detect traffic sign. If we change the data, maybe it can detect different target.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    - During training stage, I only augment some training sample to train my model. In validation and test stage, I used the trained model to predict accuracy. I haven't merge training sample with validation or test samples. The datasets are totally separated. But after apply the trained model to validation and test data we can also get 97.8% and 94.4% accuracy. It really learned something.

Here is the Training and validation accuracy diagram.

![alt text][image13] 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

| Children crossing     | No entry     	       | Yield                |  Stop                | Priority road        |
|:---------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| ![alt text][image14]  | ![alt text][image15] | ![alt text][image16] | ![alt text][image17] | ![alt text][image18] |

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing     | Traffic signals   							| 
| No entry     			| No entry 										|
| Yield					| Yield											|
| Stop	      		    | Stop					 				        |
| Priority road			| Priority road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the predictions.

![alt text][image19]

We can found that it misclassified children crossing as traffic signals. Maybe they are all triangle and the white area inside the triangle are similar.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

| Layer                      | Feature map	        | 
|:--------------------------:|:--------------------:| 
| convolution 1              | ![alt text][image20] |
| convolution 1 activation   | ![alt text][image21] |
| convolution 1 max pooling  | ![alt text][image22] |

In convolution 1, we can see it can probably detect the shape.

In convolution 1 activation, we can feel the shape is a little more clear.

In convolution 1 max pooling, since we use max pooling to extract important portion. But the resolution of the image is not so high, we can find that the effect is not that good. This maybe the cause to let the model misclassified children crossing as traffic signs.
