"""Test model."""

import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
#import lineDetector
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pylab as plt
#from lineDetector import canny_edge_detector, create_coordinates, region_of_interest, display_lines, average_slope_intercept
import numpy as np
from model import *

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

transformations = transforms.Compose(
    [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(10, 0.0010)
set_speed = 20
controller.set_desired(set_speed)

# MAX_SPEED = 15
# MIN_SPEED = 10
# speed_limit = MAX_SPEED

#image_buf = np.zeros((1, 59, 255, 3))
#state_buf = np.zeros((1,4))

#font                   = cv2.FONT_HERSHEY_SIMPLEX
#bottomLeftCornerOfText = (10,10)
#fontScale              = 0.3
#fontColor              = (255,0,0)
#lineType               = 2


def drawDirection(img, steeringAngle):

    centerOfImg = (150, 150)
    axesLength = (25, 100)
    angleRot = 0
    startAngle = 0
    endAngle = int(90 * steeringAngle / 1.2)
    colorRed = (0, 0, 255)
    colorGreen = (0, 255, 0)
    thickness = 2

    #image = cv2.ellipse(image, center_coordinates, axesLength, anglerot, startAngle, endAngle, color, thickness)

    if steeringAngle > 0:
        center_coordinates = (150 + 25, 140)
        startAngle = 0
        angleRot = 180
        poly = cv2.ellipse2Poly(center_coordinates, axesLength, angleRot,
                                startAngle, endAngle, 10)
        polyarray = np.array(poly)
        polyCasted = np.int32([polyarray])

        newImg = cv2.polylines(img,
                               polyCasted,
                               False,
                               colorRed,
                               thickness,
                               lineType=cv2.LINE_AA)
        #Añadimos una flecha al final

        finalPoint = tuple(poly[-1])
        '''
        fiveFinalPoint = tuple(poly[-2])
        print(finalPoint, fiveFinalPoint)
        newImg = cv2.arrowedLine(newImg, fiveFinalPoint, finalPoint, colorRed,
                                 2)
                                 '''
        newImg = cv2.circle(newImg, finalPoint, 3, colorRed, -1)

    # newImg = cv2.ellipse(img, center_coordinates, axesLength, angleRot,
    #                     startAngle, endAngle, colorRed, thickness)
    elif steeringAngle == 0:
        #image = cv2.arrowedLine(image, start_point, end_point, color, thickness)
        newImg = cv2.arrowedLine(img, centerOfImg, (215, 330 - 50), colorRed,
                                 thickness)
    elif steeringAngle < 0:
        center_coordinates = (150 - 25, 140)
        poly = cv2.ellipse2Poly(center_coordinates, axesLength, angleRot,
                                startAngle, endAngle, 10)
        polyarray = np.array(poly)
        polyCasted = np.int32([polyarray])

        newImg = cv2.polylines(img,
                               polyCasted,
                               False,
                               colorRed,
                               thickness,
                               lineType=cv2.LINE_AA)
        finalPoint = tuple(poly[0])
        newImg = cv2.circle(newImg, finalPoint, 3, colorRed, -1)

    return newImg






@sio.on('telemetry')
def telemetry(sid, data):

    if data:

        # angle = data["steering_angle"].replace(',', '.')
        acelerador = data["throttle"].replace(',', '.')
        vel = data["speed"].replace(',', '.')

        # The current steering angle of the car

        #steering_angle = float(angle)

        # Aceleración actual del vehículo
        throttle = float(acelerador)

        # Velocidad actual del vehículo.
        speed = float(vel)
        print('Current speed', speed)

        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        image_arrayNP = np.array(image.copy())
        image_arrayFrontal = image_arrayNP[:, :, ::-1]
        image_arrayCroped = image_arrayNP[65:-25, :, :]

        # transform RGB to BGR for cv2
        image_arrayCV = image_arrayCroped[:, :, ::-1]
        image_arrayTransformada = transformations(image_arrayCV)

        image_tensor = torch.Tensor(image_arrayTransformada)
        image_tensor = image_tensor.view(1, 3, 70, 320)
        image_tensor = Variable(image_tensor)

        #AQUII

        steering_angle = model(image_tensor).view(-1).data.numpy()[0]
        #print(model(image_tensor).view(-1).data)

        #throttle = modelLinear(steering_angle).view(-1).data
        #speed = controller.update(float(speed))

        # ----------------------- Improved by Siraj ----------------------- #
        # global speed_limit
        # if speed > speed_limit:
        #     speed_limit = MIN_SPEED
        # else:
        #     speed_limit = MAX_SPEED

        throttle = 1.2 - steering_angle**2 - (speed / set_speed)**2

        # ----------------------- Improved by Siraj ----------------------- #

        angulo_coma = str(steering_angle).replace('.', ',')
        acel_coma = str(throttle).replace('.', ',')
        send_control(angulo_coma, acel_coma)
        print("Angulo de giro: {} | Acelerador: {}".format(
            steering_angle, throttle))

        #Detector de lineas

        #lanes = getLanes(image_arrayFrontal)

        #combo_image = cv2.addWeighted(image_arrayFrontal, 0.8, line_image, 1, 1)
        #cv2.imshow("Road Lanes", combo_image)
        if image_arrayNP is not None:
            img = drawDirection(image_arrayNP, steering_angle)

            img = cv2.resize(img,
                            None,
                            fx=1.0,
                            fy=1.0,
                            interpolation=cv2.INTER_LINEAR)
            topText = 10
            interlineado = 10

            cv2.putText(img, 'Angulo de giro: ' + str(round(steering_angle, 2)),
                        (10, topText), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),
                        1)
            cv2.putText(img, 'Acelerador: ' + str(round(throttle, 2)),
                        (10, topText + interlineado), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 255), 1)
            cv2.putText(img, 'Velocidad: ' + str(round(speed, 2)),
                        (10, topText + 2 * interlineado), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0,255,255), 1)
            cv2.putText(img, 'Velocidad Set: ' + str(round(set_speed, 2)),
                        (10, topText + 3 * interlineado), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 255), 1)

            #combo_image = cv2.addWeighted(img, 0.8, lanes, 1, 1)

            
            cv2.imshow('Camara Frontal', img)
            cv2.waitKey(1)

        # Guardar frame

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            img.save('{}.jpg'.format(image_filename))

    else:
        # NOTE: NO EDITAR
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer",
             data={
                 'steering_angle': steering_angle,
                 'throttle': throttle
             },
             skip_sid=True)


if __name__ == '__main__':
    """Testing phase."""
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help=
        'Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Definición del modelo.
    # model = LeNet()
    model = NetworkNvidia()

    
    # Verificación de que el modelo es el mismo que la versión local de Pytorch.
    try:
        checkpoint = torch.load(args.model,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    except KeyError:
        checkpoint = torch.load(args.model,
                                map_location=lambda storage, loc: storage)
        model = checkpoint['model']

    except RuntimeError:
        print("==> Please check using the same model as the checkpoint")
        import sys
        sys.exit()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Utilizo Flask utilizando el socket-io como Middleware
    app = socketio.Middleware(sio, app)

    # Despliegue como un evento en el puerto 4567 de un WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
