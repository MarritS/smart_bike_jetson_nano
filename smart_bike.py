#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:30:34 2021

@author: marrit
contact: marritschellekens@yahoo.com
"""

"""smart_bike.py

This script starts the smart_bike code, which detects upcoming vehicles using Yolo and a custom tracking algorithm. 

"""

#Use the Camera device
CAMERA = False
#Write analysed frames to file. This makes the program run significantly slower.
WRITE_TO_FILE = False
#Video file to use if CAMERA is set to False
VIDEO_FILE = "videos/d_raw.mp4" 
#The simulated fps when analyzing a video file. Ignored when using camera. 
SIMULATED_FPS = 6
#Shows the matches made by the tracker, and prints the metric values. Press space to go to the next match. 
SHOW_MATCHES = False
WINDOW_NAME = "tracking"

import cv2
from utils.camera import Camera
import os
import time
from utils.display import open_window, set_display, show_fps
import CentroidTracker
import multiprocessing as mp
import pycuda.autoinit #This is needed for intilazing CUDA driver
#Originally the idea was that another file could be written that contained functions for my windows laptop. This could be imported here as an alternative to Jetson_functions. This idea never really came to fruition, but the structure remains. 
import Jetson_functions as hw_functions
import helper_functions as functions
from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import get_input_shape, TrtYOLO

 
def loop_and_detect(cam, trt_yolo, classes, writer):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    framecounter = 0

    frameskip = round(6/SIMULATED_FPS)
    hw_func = hw_functions.jetson()
    tracker = CentroidTracker.CentroidTracker(m_DEBUG=SHOW_MATCHES)
    
    start = time.time()
    while True:
        print('frame:' + str(framecounter))
        
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        if CAMERA:
            tic, fps = update(tic, fps, img, trt_yolo, classes, tracker, hw_func,  writer)
        elif framecounter % frameskip == 0:
            tic, fps = update(tic, fps, img, trt_yolo, classes, tracker, hw_func,  writer)
        
        
        framecounter+=1
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display('test', full_scrn)
        end = time.time()
        while (end-start < framecounter/SIMULATED_FPS) and CAMERA==False:
            end = time.time()

 
def update(tic, fps, img, trt_yolo, classes, tracker, hw_func, writer):
	tic, fps = functions.update_fps(tic, fps)
	frame_detection, frame_tracking = functions.detect_and_track(img, trt_yolo, classes, tracker,hw_func)
	frame_tracking = show_fps(frame_tracking, fps)
	frame_tracking = cv2.resize(frame_tracking, (800, 600))
	cv2.imshow(WINDOW_NAME, frame_tracking)
	if WRITE_TO_FILE:
        	writer.write(frame_tracking)
	return tic, fps


def main():
    #You don't need to pass arguments. I kind of hacked around that because I was too lazy to completely rewrite this part. 
    args = hw_functions.parse_args(VIDEO_FILE, CAMERA)
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
        
    cls_dict = get_cls_dict(args.category_num)
    h, w = get_input_shape(args.model)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    VIDEO_NAME = "/home/marrit/Videos/output.mp4"
    frame_width = 800
    frame_height = 600

    #A quick hack to prevent a file being created when we don't want to save the files. 
    if WRITE_TO_FILE:
        writer = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), SIMULATED_FPS, (frame_width, frame_height))
    else:
        writer = None
    
    
    open_window(  
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, cls_dict, writer)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mp.freeze_support()
    main()
