#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:30:34 2021

@author: marrit
"""

"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

import cv2
from utils.camera import Camera



import os
import time

from utils.display import open_window, set_display, show_fps

import CentroidTracker
import multiprocessing as mp
import pycuda.autoinit #This is needed for intilazing CUDA driver
import Jetson_functions as hw_functions
import helper_functions as functions
from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import get_input_shape, TrtYOLO
import shared_space_detector
import roadTypeTracker
#import VariableWriter



CAMERA = False
WRITE_TO_FILE = True
VIDEO_FILE = "/home/marrit/Videos/a.mp4" 
SIMULATED_FPS = 6
FPS_SS = 0.3
WINDOW_NAME = "tracking"




 
def loop_and_detect(cam, trt_yolo, classes, writer):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    framecounter = 0

    frameskip = round(6/SIMULATED_FPS)
    frameskip_ss = round(6/FPS_SS)
    hw_func = hw_functions.jetson()
    tracker = CentroidTracker.CentroidTracker()
    tracker_ss = roadTypeTracker.roadTypeTracker()
    
    
    start = time.time()
    while True:
        print('frame:' + str(framecounter))
        
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        #if framecounter % frameskip_ss == 0:
            #separated_bikepath = tracker_ss.update(img)
            
        
        if CAMERA:
            tic, fps = update(tic, fps, img, trt_yolo, classes, tracker, hw_func, False, writer)
        elif framecounter % frameskip == 0:
            tic, fps = update(tic, fps, img, trt_yolo, classes, tracker, hw_func, False, writer)
        
        
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

 
def update(tic, fps, img, trt_yolo, classes, tracker, hw_func, sep_path, writer):
    tic, fps = functions.update_fps(tic, fps)
    if not sep_path:
        frame_detection, frame_tracking = functions.detect_and_track(img, trt_yolo, classes, tracker,hw_func)
        frame_tracking = show_fps(frame_tracking, fps)
        frame_tracking = cv2.resize(frame_tracking, (800, 600))
    else:
        frame_detection = img
        frame_tracking = functions.drawTracking_separated(img)
    cv2.imshow(WINDOW_NAME, frame_tracking)
    if WRITE_TO_FILE:
        writer.write(frame_tracking)
    return tic, fps

def main():
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
    #When using camera, and you want the file saved, adjust size parameters
    #VIDEO_NAME = "/media/marrit/E61E-23E2/output_" + 'TIME' + ".mp4"
    #VIDEO_NAME = "/home/marrit/project/tensorrt_demos/output_" + 'TIME' + ".mp4"
    VIDEO_NAME = "/home/marrit/Videos/output.mp4"
    frame_width = 800
    frame_height = 600

    #A quick hack to prevent a file being created when we don't want to save the files. 
    if WRITE_TO_FILE:
        writer = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), SIMULATED_FPS, (frame_width, frame_height))
    else:
        writer = None
    #varWriter = VariableWriter.VariableWriter(SIMULATED_FPS, writer)
    
    
    #writer = cv2.VideoWriter('output.mp4', fourcc, SIMULATED_FPS, (800, 600), True)
    open_window(  
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, cls_dict, writer)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mp.freeze_support()
    main()
