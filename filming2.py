#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:22:45 2021

@author: marrit
"""

import cv2
import Jetson_functions as hw_functions
import os
from utils.camera import Camera

args = hw_functions.parse_args('nothing', True)
if args.category_num <= 0:
    raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
if not os.path.isfile('yolo/%s.trt' % args.model):
    raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

capture = Camera(args)
if not capture.isOpened():
    raise SystemExit('ERROR: failed to open camera!')
 
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
videoWriter = cv2.VideoWriter('/home/marrit/Videos/test.avi', fourcc, 30.0, (640,480))
 
while (True):
 
    frame = capture.read()
    
     
    if frame is not None:
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('video', frame)
        videoWriter.write(frame)
 
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
videoWriter.release()
 
cv2.destroyAllWindows()