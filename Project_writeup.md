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

[image1]: ./Output/label_distribution.png "Dataset label distribution"
[ref1]: (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html)
[ref2]: (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
[ref3]: (https://navoshta.com/traffic-signs-classification/)
[image2]: ./Output/hist_equalization.png "Effect of implementing histogram equalization"
[image3]: ./Output/accuracy.png "Accuracy history"
[image4]: ./Output/testimages.png "10 test images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

After loading the pickled data train.p, valid.p and test.p which represent training dataset, validation dataset, and testing dataset, respectively, dataset size and image size are calculated mainly based on the shape:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes of training, validation, and testing dataset are distributed.

![alt text][image1]

Majority of the traffic signs in the dataset are among 1-13 (speed limits, No passing, Right-of-way at the next intersection, Priority road, yield, etc...).

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Images from the dataset are color in RGB of size 32x32x3. According to some open discussion on the Internet, training a network with traffic sign images of 3 color channels does not yield a significantly better prediction performance with those of only 1 color channel. To reduce the number of parameters in the network, RGB channels of each image are combined to a single Y channel based on YUV color coding (similar to grayscale), using opencv function. 

Next, I applied the histogram equalization to the one channel image. 
![alt text][image2]
On the left, an original traffic sign image (#34481 in the training dataset) is shown. The whole image is dark and it is hard to identify the traffic sign type by eye. Referring to open discussion, histogram equalization (![alt text][ref1]) is very helpful to improve the contrast of the image. The same traffic sign after applying the histogram equalization is shown on the right. It is lighter while keeping all the edge/line/shape details for future classification.

Finally, I divided the pixel values by 255 and convert them to floating numbers so that the image pixels are normalized between 0 and 1 to reduce the range of weight/biases to be trained. Training, validation, and testing dataset are all prepocessed in the same way. I didn't use extra dataset for training.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The structure of the model is based on LENET basic CNN which connect layers in series only, a paper titled "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun (![alt text][ref2]) which introduces fusing multi-layer output together, and idea from (![alt text][ref3]) on considering combining three layers with the same size 4x4 neuron output together for better performance.  

My final model consisted of the following layers:

| Layer         		|     Description	        							| 
|:---------------------:|:-----------------------------------------------------:| 
1| Input         		| 32x32x1 normalized image   							| 
2| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 			|
3| RELU					|														|
4| Max pooling	      	| 2x2 stride,  outputs 16x16x32 						|
5| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    		|
6| RELU					|														|
7| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				    		|
8| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x256     		|
9| RELU					|														|
10| Max pooling	      	| 2x2 stride,  outputs 4x4x256 				    		|
11| Flatten				| Combine max pooling of layer 4's output(4x4 stride), 	| 
                                  max pooling of layer 7's output(2x2 stride), 	|
                              and layer 10 , output 5632x1    					|
12| Dropout				| dropout rate 0.3 										|
13| Fully connected		| output 1024x1 										|
14| RELU				| 				 										|
15| Dropout				| dropout rate 0.3 										|
16| Fully connected		| output 256x1 											|
17| RELU				| 				 										|
18| Dropout				| dropout rate 0.5 										|
19| Fully connected		| output 43x1 											|

The main part includes three convolution layers followed by RELU activation and maxpooling. 1st, 2nd, 3rd convolution layers after maxpooling are layer 4, 7 and 10. Their size are 16x16x32, 8x8x64, 4x4x256, respectively. At layer 11, to combine these three layers that capture different levels of details in the image with the same size, additional maxpooling of 4x4 stride, 2x2 stride are applied on output of layer 4 and output of layer 7. The whole size of flattening at layer 11 are 4x4x(32+64+256)=5632. Next is a total of 2 fully connected layers (layer 13,16) with RELU activation and a final fully connected layer to convert to onecoding of size 43. Dropout layers are added to reduce overfitting.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used ADAM optimizer with batch size of 128. The loss function is the mean of cross entropy between final outputs (logits) and labels (onecoded y_train). Since the model is deep and contains a large quantity of weights and biases parameters, the maximum accuracy trained using this model should be satisfying. Thus, the initial learning rate is set to be small (0.0001) to achieve a finer training. Also, exponential decaying on learning rate (beta = 0.95) every 200 evaluation are implemented as well. Since there are about (34799/128 = 272) calls on every epoch, so roughly the learning rate is decaying every epoch. I run 101 epochs. For evaluating on validation and test images, no dropout (dropout rate = 0) is implemented. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

This model has the property of combining pooling results of three convolution layers at different stages. The first layer could be detecting basic elements like edges, and second, third convolution may be able to tell the sub-feature of the traffic sign (for example, is it a triangle or circle sign) and the detailed content like text or icon inside. 

I run the training and evaluation on a local machine (i7-6820HQ + Nvidia quadro m1200, cm5.0). After running in tensorflow gpu mode for 101 epochs, my final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.968
* test set accuracy of 0.954

Below shows the history of accuracy evaluated using training dataset (calling training accuracy) and accuracy evaluated using validation dataset (calling validation accuracy). Training accuracy is always greater than validation accuracy and at the end very close to 1. This means the proposed model is well trained based on images in the training dataset. I found the result is satisfied with this current model because both validation accuracy and accuracy on test images are above 0.93. However, validation accuracy converges to about 0.97 and doesn't converge to 1. It could be that more epoch needs to be trained, or learning rate is too small due to exponential decaying, or most likely overfitting. There could be more techniques and freedom to improve the validation accuracy such as implementing L1/L2 regularization, tuning/adding dropout layers, using more training data, etc. 

![alt text][image3]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. 

I chose 10 random German traffic signs from the testing dataset. Since testing dataset is only used for final evaluation, my model's parameters are not influenced by test images and I think it is valid to show the model result using all trained parameters. The index numbers of these 10 test images are also showed. The three speed limits (3rd, 7th, 9th) and the 2nd image may be hard to detect because they are either too dark or the icon inside is too blur.  

![alt text][image4] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

  | Image			        		| Prediction	        	| Correct?
  |:-------------------------------:|:-------------------------:| 
 1| 30 km/h     					| 30 km/h   				| Y
 2| Road work     					| Road work 				| Y
 3| 120 km/h	      				| 120 km/h					| Y
 4| Yield							| Yield						| Y
 5| Dangerous curve to the right 	| Slippery Road				| N
 6| Keep right                   	| Keep right 				| Y
 7| 70 km/h     					| 70 km/h   				| Y
 8| Ahead only						| Ahead only				| Y
 9| 100 km/h	      				| 100 km/h					| Y
10| Yield							| Yield						| Y

The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. This compares favorably to the accuracy on the test set of 0.954. However, for the only wrong detection, the model guesses it is "Slippery Road" (label 23) instead of "Dangerous curve to the right" (label 20). I have no idea why because these two kinds of traffic sign are not very alike except that the left strip on the "Slippery Road" sign is similar to the right turn like "Dangerous curve to the right" sign. Maybe the model is still not able to distinguish these two types for sure.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the final cell of the Ipython notebook. The top five most likely traffic sign type in enumeration is listed below for all images.
  |Top five traffic sign type 	|
  |:---------------------------:|
 1| 1,  2,  5,  6,  4   		|
 2|25, 22, 29, 24, 19			|
 3| 8,  7,  4, 15,  5			|
 4|13, 35,  5, 12, 33			|
 5|23, 20, 29, 10, 25			|
 6|38, 34, 10, 36, 12			|
 7| 4,  8, 15,  0,  1			|
 8|35, 33, 36,  3, 34			|
 9| 7,  5,  8,  4, 10			|
10|13, 35, 33, 12, 39			|

For all the images except the wrongly detected one (5th), the top first probability is almost equal to 1 and the rest 2-5 most likely probability are on the order of 10E-9 to 10E-21, meaning the model is very certain about the classification and the detection results confirm that it is correct. For the 5th image, top five soft max probabilities are:

| Probability         	|     Prediction	        						| 
|:---------------------:|:-------------------------------------------------:| 
| .492         			| [23] Slippery Road   								| 
| .367     				| [20] Dangerous curve to the right 				|
| .076					| [29] Bicycle crossing								|
| .022	      			| [10] No passing for vehicles over 3.5 metric tons |
| .018				    | [25] Road work									|

The second most likely type is the correct answer. This model has the potential of detecting it out correctly, but for this particular image, the model needs to be modified or trained more to achieve better performance. Another explanation is that the model is already overfitting and may lead to wrong detection for images that are not used for training before.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


