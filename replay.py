# -*- coding: utf-8 -*-
"""
Created on Tue May 11 16:19:23 2021

@author: marri
"""

import pickle
from utils.camera import Camera
import pycuda.autoinit #This is needed for intilazing CUDA driver
import Jetson_functions as hw_functions
import cv2
import pycuda.autoinit 
import helper_functions as functions
import time
import math
import Serial
import os

FRAMERATE = 5

INPUT_FILE="/home/marrit/Videos/00124.mp4"

myDict = pickle.load( open( "00124_save.p", "rb" ) )

hw = hw_functions.jetson()

frame_nrs = myDict['frame_nrs']
rects = myDict['rects']
directions = myDict['directions']


#vs = cv2.VideoCapture(INPUT_FILE)
#hw.init_cam(INPUT_FILE, False)
totalcnt = 0
STARTING_FRAME = 1200

args = hw_functions.parse_args(INPUT_FILE)
if args.category_num <= 0:
	raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

cam = Camera(args)
if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

def perform_output(rects, directions):
    max_size = 0
    
    for key, rect in rects.items():
        if direction[key] == 1:
            d_x = rect[2] - rect[0]
            d_y = rect[3] - rect[1]
            size = math.sqrt(d_x * d_y)
            if size>max_size:
                max_size = size
            print(size)
    
    if (max_size > 250):
        output = 3
    elif (max_size > 120):
        output = 2
    elif (max_size > 0):
        output = 1
    else:
        output = 0
        
    Serial.write_read(str(output))


while True:
    totalcnt += 1
    frame = cam.read()
    if frame is None:
        break
    if totalcnt == frame_nrs[0]:
        frame_nrs.pop(0)
        rect = rects.pop(0)
        direction = directions.pop(0)
        
        if totalcnt >= STARTING_FRAME:
            frame_tracking = functions.drawTracking(frame, rect, direction)
            frame_tracking = cv2.resize(frame_tracking,(800, 600))
            cv2.imshow('tracking', frame_tracking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
               break
            if len(frame_nrs) == 0:
                break
            
            perform_output(rect, direction)
            time.sleep(1./5)
            


Serial.close()
        
