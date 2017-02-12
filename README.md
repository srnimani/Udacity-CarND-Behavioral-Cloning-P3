# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files for Udacity - Behavioral Cloning P3 project work.

Initial Committ on 3rd Feb 2016. Modified files loaded on 11th Feb 2017

Model inspiration and some ideas came from the work done by Vivek Yadav and Matt Harvey..

https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.2oj0h0bon

https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.f1chrdhkd

Originally used NVidia CNN model, as described in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

which did not work too well

Used the default data from Udacity to train most of the model and once it started working well for both original tracks added my own data to smoothen straightline drives (it still swaggers a bit). Added a few laps of of straightline drives at fast pace, and then added a few recovery data around the edges and corners. Additional data included 3500 images. Could have added a lot more data if I had better control over the mouse. Did not train on track 2 at all.  

Sample Images:

Driving Straight:

https://cloud.githubusercontent.com/assets/10118232/22862239/1562fb48-f151-11e6-8626-4fb17b9446c7.jpg

https://cloud.githubusercontent.com/assets/10118232/22862241/1c2a57fa-f151-11e6-8ea1-d8770e087818.jpg

https://cloud.githubusercontent.com/assets/10118232/22862250/4750b47e-f151-11e6-883f-ae7a8288dcf7.jpg

3 Camera Images (Left, center and right) for the same scene, used a traslation of 0.25 to add / subtract to the angle of left and right cameras as if they came from the center camera. In the following 3 images, the steering angle was 0 with respect to the center camera.

https://cloud.githubusercontent.com/assets/10118232/22862337/98c87ef8-f152-11e6-9e59-d1c63a694c7d.jpg

https://cloud.githubusercontent.com/assets/10118232/22862329/7fe3c424-f152-11e6-8027-7f0adc4d3af8.jpg

https://cloud.githubusercontent.com/assets/10118232/22862345/ae0a60e2-f152-11e6-81da-320782e1e4e7.jpg

Recovery recording:



Latest simulator data track 2 does not work, I guess due to lane lines and very difficult terrian. Needs some more work.

Used Jupyter Notebook

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


## Details About Files In This Directory

### `drive.py`

Drive.py is the stimulator for the model to drive the car around the track.. Takes the image from the simulator and feeds to the model to get the desired output steering angle, and then processes the desired steering and throttle values to drive the
car in the simulator.

Usage of `drive.py` requires the saved the trained model as an .json file, and then used with the model using this command:

```sh
python drive.py model.json
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

### `model.json`

model.json is the model used by Drive.py to drive the car around the track.. Created by the Jupyter note book, 'Behavioral-cloning-P3-Submit.ipynb' after training using the images. 

### `model.h5`

model.h5 is the file containg the weights for the model used by Drive.py to drive the car around the track.. Created by the Jupyter note book, 'Behavioral-cloning-P3-Submit.ipynb' after training using the images. 

### `Behavioral-cloning-P3-Submit.ipynb`

Jupyter note book, containing the actual code for training and generating the model and the weights.
