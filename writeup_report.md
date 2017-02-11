#**Behavioral Cloning** 

##Writeup of work done for Behavioral Cloning Project of Udacity, by Mani Srinivasan

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.ipynb is the Jupyter notebook that contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I have also added the model.py code for sake of completion.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 139-169) 

The model includes ELU layers to introduce nonlinearity (143-168), and the data is normalized in the model using a Keras lambda layer (code line 140). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 143-168). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 88-104). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, with a learning rate of 0.0001 (model.py lines 170-173).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one in an article by nVidia. http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf.  But having worked on it for close to a week and not finding a good solution, I switched to a CNN model described in another paper by Vivek Yadav, in https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1g3gw99kj... This model immediately started yielding the results I was struggling to get initially.  This model uses a set of filters and convolutional networks, maxpooling, drop of layers and finally fully connected laters. The first layer is 3 1X1 filter, which has the effect of transforming the color space of the images. Using 3 1X1 filters allows the model to choose its best color space. This is followed by 3 convolutional blocks each comprised of 32, 64 and 128 filters of size 3X3. These convolution layers were followed by 3 fully connected layers. All the convolution blocks and the 2 following fully connected layers had exponential relu (ELU) as activation function. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 25% of the samples as validation data. To address overfitting, I used several image processing techniques including, flipping, image augmentation by varying the brightness randomly, shifting the images to the left and right by a few pixels, using alterante camera images with modified steering angle etc..

It took a while to get the car go around the track without falling off. Till the car went aroung track 1 for a few laps, I did not add any data on my own.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. As the weaving of was a little dissconcerting, I modifid the drive.py to adjust the steering angle by averaging over multiple frames and also lowering the speed so that the car does not go off the edges.

####2. Final Model Architecture

The final model architecture (model.py lines 143-168) consisted of a convolution neural network with the following layers and layer sizes.


    - Normalization Layer using Lamda function.     
    - 3 Sets of CNN layers with Maxpooling and dropout to prevent overfitting 
      - Set 1:
        Convolution - 32x3x3 followed by ELU activation
        Convolution - 32x3x3 followed by ELU activation
        Maxpooling - 2x2 filter
        Dropout - 50%
      
      - Set 2:
        Convolution - 64x3x3 followed by ELU activation
        Convolution - 64x3x3 followed by ELU activation
        Maxpooling - 2x2 filter
        Dropout - 50%
      
      - Set 3:
        Convolution - 128x3x3 followed by ELU activation
        Convolution - 128x3x3 followed by ELU activation
        Maxpooling - 2x2 filter
        Dropout - 50%
      
      - Fully Connected Layers
        FC 512 with ELU activation
        FC 64 with ELU activation
        FC 16 with ELU activation
        FC 1 with ELU activation (Output)
      
####3. Creation of the Training Set & Training Process

For a large part of the work, I used the Udacity provided data to train and model the network. This is due to the difficulty I faced in controlling the car with a normal mouse. Later, to capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from edges without falling off. After the collection process, I had about 11540 number of data points. I would have collected more to get a much smoother drive, but for the difficulty in controlling the car with the mouse, adjusting the speed and recording at the same time on my Macbook Pro. 

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as more number of epochs did not yield any better results. I used an adam optimizer with a learning rate of 0.0001.
