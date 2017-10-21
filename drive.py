import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

class SimplePIController:
    """
    Simple cruise controller - increase or decrease throttle to keep desired speed on same level
    Modified to use limited number of history values to avoid error value uncontrollable increase.
    """

    def __init__(
        self,
        Kp, # Multiplier for current speed deviation
        Ki, # Multiplier for integral speed deviation
        Ks, # Number of history steps used to integral speed deviation calculation
        target_speed = 0.0): # Target speed
        self.Kp = Kp
        self.Ki = Ki / Ks
        self.Ks = Ks
        self.target_speed = target_speed
        self.error = 0.0
        self.integral = [0.0]

    def set_target_speed(self, target_speed):
        self.target_speed = target_speed

    def update(self, measurement):
        # proportional error
        self.error = self.target_speed - measurement

        # integral error
        self.integral += [self.error]
        sz = len(self.integral)
        if sz > self.Ks:
            self.integral = self.integral[sz - self.Ks:] # Queue implementation

        return self.Kp * self.error + self.Ki * sum(self.integral)


controller = SimplePIController(0.1, 0.2, 100, 30.0) # Initialize cruise controller

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = data["steering_angle"] # The current steering angle of the car
        throttle = data["throttle"] # The current throttle of the car
        speed = data["speed"] # The current speed of the car
        imgString = data["image"] # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        measures_predict = model.predict(image_array[None, :, :, :], batch_size = 1) # Predict model parameters
        measures_predict = measures_predict[0]
        steering_angle = float(measures_predict[0]) # Retrieve predicted steering angle

        if len(measures_predict) >= 2: # In case model use more that 1 parameter, 2nd must be predicted speed level
            target_speed = (float(measures_predict[1]) * 100.0) + 15.0 # Retrieve predicted speed level
            controller.set_target_speed(target_speed) # Update cruise controller with neew traget speed value
        else:
            target_speed = float(speed) # In case model use only 1 parameter, target speed is not changed.

        throttle = controller.update(float(speed))

        print("St Angle: {:.5f}, Tg Speed: {:.5f}, Throttle: {:.5f}".format(steering_angle, target_speed, throttle))
        send_control(steering_angle, throttle) # Send to simulator

        if exp_image_folder != '':
            # Save frame
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(exp_image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default='model.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model_file = args.model
    exp_image_folder = args.image_folder

    #model_file = 'model.h5'
    #exp_image_folder = ''

    # Check that model Keras version is same as local Keras version
    f = h5py.File(model_file, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(model_file) # Load Keras model

    if exp_image_folder != '':
        # Create folder for images
        print("Creating image folder at {}".format(exp_image_folder))
        if not os.path.exists(exp_image_folder):
            os.makedirs(exp_image_folder)
        else:
            shutil.rmtree(exp_image_folder)
            os.makedirs(exp_image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
