import argparse
import base64
import json
import cv2

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import math

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None

steering_history_array_length = 1
steering_history = [0]*steering_history_array_length
previous_steering = 0
steering_history_index = 0


def resizeImage(image):
    # Preprocessing image files
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64, 64), interpolation=cv2.INTER_AREA)    
    return image 


@sio.on('telemetry')
def telemetry(sid, data):
    
    # Additional global variables 
    
    global steering_history_index
    global steering_history

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_pre = np.asarray(image)
    image_array = resizeImage(image_pre)
    
    transformed_image_array = image_array[None, :, :, :]
    
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # Additional Code to ensure smooth steering
    
    new_steering_angle = (steering_angle + np.sum(steering_history)) / (1+steering_history_array_length)
    steering_history[steering_history_index] = new_steering_angle
    steering_history_index += 1
    if steering_history_index >= steering_history_array_length:
        steering_history_index = 0

    # Control speed based on steering angle
    
    throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35) 
        
    # Code for failsave lest car stalls on slopes/ difficult conditions:
    
    if float(speed) < 10.:
        throttle = 0.8

    print(steering_angle, throttle)
    send_control(new_steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
