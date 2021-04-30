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
from utils.yolo_with_plugins import get_input_shape, TrtYOLO
import cv2


from utils.camera import add_camera_args, Camera
from utils.visualization import BBoxVisualization

import os
import time
import argparse


import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np



from utils.display import open_window, set_display, show_fps


import CentroidTracker
from utils.yolo_classes import get_cls_dict




WINDOW_NAME = 'TrtYOLODemo'
CAMERA = False
VIDEO_FILE = "/home/marrit/Videos/city_moving2.mp4" 
SIMULATED_FPS = 7

colors = np.zeros([3,3])
colors[2] = [0, 0, 255]
colors[1] = [50, 50, 50]
colors[0] = [255, 0, 0]

tracker = CentroidTracker.CentroidTracker()
 


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, default='yolov4-csp-256',
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')

    if CAMERA:
        args = parser.parse_args(['--onboard', 0, '--width', '800', '--height', '600'])
        #args = parser.parse_args(['--width', 800])
        #args = parser.parse_args(['--height', 600])
    else:
        args = parser.parse_args(['--video', VIDEO_FILE])
    return args

def drawTracking(img, objects, boxes, rects):
    ids = []
    boxesList = boxes.tolist()
    rectcounter = 0

    for inputCentroid in boxesList:
        rectcounter=0
        directions = tracker.returnDirections()

        for key, value in objects.items():
            if(value==inputCentroid).all():
                rectcounter +=1
                if (rectcounter > 1):
                    print(' Error')
                index = boxesList.index(inputCentroid)
                rect = rects[index]
                direction = directions[key]
                p1 = (int(rect[0]), int(rect[1]))
                p2 = (int(rect[2]), int(rect[3]))
                rectcounter +=1
                cv2.rectangle(img, p1, p2, colors[direction+1], 2)
                cv2.putText(img, str(key), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 2)
                break;
    return img


def loop_and_detect(cam, trt_yolo, conf_th, vis, writer):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    framecounter = 0
    frameskip = round(30/SIMULATED_FPS)
    

    
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break
        
        if CAMERA:
            img = detect(img, trt_yolo, conf_th, vis, fps, tracker)
            cv2.imshow(WINDOW_NAME, img)
            writer.write(img)
        elif framecounter % frameskip == 0:
            img = detect(img, trt_yolo, conf_th, vis, fps, tracker)
            cv2.imshow(WINDOW_NAME, img)
            writer.write(img)
        
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        framecounter+=1
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        

def detect(img, trt_yolo, conf_th, vis, fps, tracker):
     boxes, confs, clss = trt_yolo.detect(img, conf_th)

     img_detection = img.copy()
     img_tracking = img.copy()
     poststamps=[]
     rects=[]
     entering = []
     leaving = []
     inputCentroids = np.zeros((len(boxes), 2), dtype = "int" )
     for (i, (startX, startY, endX, endY)) in enumerate(boxes):
         cX = int((startX + endX) / 2.0)
         cY = int((startY + endY) / 2.0)
         inputCentroids[i] = (cX, cY)

         x = int(startX)
         y = int(startY)
         w = int(endX-startX)
         h = int(endY-startY)

         leaving.append(False)
         entering.append(False)
         if x<0 :
             entering[-1] = True
             x=0
         if x > img.shape[1]-8:
             leaving[-1] = True
         if y<0:
              y = 0
         if y+h>img.shape[0]-8:
              if x>int(img.shape[1]/2):
                  leaving[-1] = True
              else:
                  entering[-1] = True
         

         poststamps.append(img[y:y+h,x:x+w])
     
     ids = tracker.update(boxes, clss, poststamps, leaving, entering)
     
     #tracked_indexes = get_tracked_indexes(objects, inputCentroids)
     
     img_detection = vis.draw_bboxes(img, boxes, confs, clss)
     img_tracking = drawTracking(img_tracking, ids, inputCentroids, boxes)
     img = show_fps(img, fps)
     return img_tracking
 
def get_tracked_indexes(objects, inputCentroids):
    ids = []
    for inputCentroid in inputCentroids:
        for key, value in objects.items():
            if (value==inputCentroid).all():
                ids.append(key)

    return ids
                
                
                


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)


    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
        

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    h, w = get_input_shape(args.model)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    #When using camera, and you want the file saved, adjust size parameters
    writer = cv2.VideoWriter('output.mp4', fourcc, SIMULATED_FPS, (1440, 1080), True)
    open_window(  
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis, writer=writer)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
