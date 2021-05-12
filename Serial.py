# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:44:48 2021

@author: marri
"""

# Importing Libraries
import serial
import time
import timeit
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)
def write_read(x):
    print('wrote: ', x)
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
   
    data = arduino.readline()
    return data

def updateVibration(num):
    #num = input("Enter a number: ") # Taking input from user
    value = write_read(num)
   # start = time.time()
   # start2 = time.time()
    #end = time.time()
    #print(value) # printing the value
    #counter = 0
    #while end-start2 < 200:
        #end = time.time()
        #print(end-start)
        #if (end-start>1):
           # value = write_read(str(0))
       # if (end-start>2):
            #value = write_read(num)
            #start = timeit.timeit()
        #counter+=1
def close():
    arduino.close()
