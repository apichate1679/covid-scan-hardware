# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import requests
import numpy as np
import imutils
import time
import cv2
import os
import GY906 as GY906
from gpiozero import Buzzer
from time import sleep
import RPi.GPIO as GPIO
from gpiozero import Servo

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFkbWluMSIsInR5cGUiOiJIQVJEV0FSRSIsImlhdCI6MTY1MzkyNzk0M30.atR5t5gi37JKyNgIqBQQD9d77quhYaMH3dCHu_51hx4"
sensor_IR  = 16
servo = Servo(25)

GPIO.setmode(GPIO.BCM) 
GPIO.setup(sensor_IR ,GPIO.IN)
led_green=18
led_red=19
GPIO.setup(led_green ,GPIO.OUT)
GPIO.setup(led_red ,GPIO.OUT)


units = 'c'

#Bus default = 1
bus = 1
#add another sensor
#bus2 = 3

#address gy906 = 0x5a
address = 0x5a

#GY906
sensor = GY906.GY906(address,bus,units)
#add another sensor
#sensor2 = GY906.GY906(address,bus2,units)
buzzer = Buzzer(17)

server_url = "https://covid-scan-backend.herokuapp.com"
def create_log(frame, temp, mask, token):
    byte_io = cv2.imencode(".JPEG", frame)
    data = io.TextIOWrapper(byte_io)

    files = {"image": data, "temp": temp, "mask": mask}
    headers = {"token": token}
    return requests.post(server_url + "/hardware/scan-log", files=files, headers=headers)

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
image_path=r"log_data/image"
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
print ("IR Sensor Ready.....")
print (" ")
ID=456798
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    temp = sensor.get_obj_temp()

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label_1 = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label_1 == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label_1, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        if GPIO.input(sensor_IR ) !=1:
            print(GPIO.input(sensor_IR ))
            if temp is not None:
                person_temp = "Temp: {0:0.1f}{1}".format(temp,units)
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                print(label)
                cv2.putText(frame, person_temp, (endX-10, endY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                create_log(frame, "{0:0.1f}".format(temp), label_1 == "Mask" if "true" else "false", TOKEN)
                if label_1 == "Mask" and int(temp) < int(37):
                    #led 1 =1
                    GPIO.output(led_green,GPIO.HIGH)
                    ID+=1
                    sleep(1)
                    servo.max()
                    sleep(5)
                    servo.mid()
                    GPIO.output(led_green,GPIO.LOW)
                    pass
                else:
                    GPIO.output(led_red,GPIO.HIGH)
                    ID+=1
                    buzzer.on()
                    sleep(5)
                    buzzer.off()
                    GPIO.output(led_red,GPIO.LOW)
            #if int(temp) > int(30) :
             #  buzzer.on()
              # sleep(5)
               #buzzer.off()
               #sleep(10)



                

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()