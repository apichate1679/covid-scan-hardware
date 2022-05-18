from gpiozero import Servo
from time import sleep

servo = Servo(25)

while True:
    servo.mid()
    sleep(1)
    servo.max()
    sleep(5)
    print("Program stopped")
