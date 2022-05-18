#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import GY906 as GY906
import time

#c = celsius
#f = Fahrenheit
#k = kelvin
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

time.sleep(1)
running = True
while(running):
    try:
        #get area temperature
        #temperature = get_amb_temp()
        #get object temperature
        temperature = sensor.get_obj_temp()
        #add another sensor
        #temperature2 = sensor2.get_obj_temp()
        if temperature is not None:
            print ('obj1 Temp={0:0.1f} {1}'.format(temperature,units))
        
        #add another sensor
        #if temperature2 is not None:
        #    print ('obj2 Temp={0:0.1f} {1}'.format(temperature2,units))
            
        time.sleep(2)
    except KeyboardInterrupt:
        running = False