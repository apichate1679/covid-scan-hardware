import RPi.GPIO as GPIO
import time

sensor = 16

GPIO.setmode(GPIO.BCM) 
GPIO.setup(sensor,GPIO.IN)
print ("IR Sensor Ready.....")
print (" ")

while True:
  if GPIO.input(sensor):
      print ("Object Detected")
      while GPIO.input(sensor):
          time.sleep(0.2)