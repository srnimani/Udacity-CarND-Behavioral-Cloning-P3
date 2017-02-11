Udacity - Self Driving Car - P3

Initial Committ on 3rd Feb 2016

Modified files loaded on 11th Feb 2017

Model inspiration from Vivek Yadav and some ideas from Matt Harvey as well..

https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.2oj0h0bon

https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.f1chrdhkd

Originally used NVidia CNN model, as described in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

which did not work too well

Used the default data from Udacity to train most of the model and once it started working well for both original tracks
added my own data to smoothen straightline drives (it stll swaggers a bit). Used low resolution, fast laps and some recovery data around the edges as well.

Modified original Drive.py to control the speed and steering control to smoothen the drive..

Latest simulator data track 2 does not work, I guess due to lane lines and very difficult terrian. Needs some more work.


Used Jupyter Notework
