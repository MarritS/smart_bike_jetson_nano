# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:03:04 2021

@author: marri
"""
VIBRATE = False

import cv2
import numpy as np
import time
import sys

if VIBRATE:
    import vibrate

def shutdown_vibration():
    if VIBRATE:
        vibrate.close()

def drawTracking_separated(img):
    frame_text = cv2.putText(img, str('Separated road, stopped detection'), (80, 80),cv2.FONT_HERSHEY_DUPLEX, 2, (50,255,50), 2 )
    return frame_text

def drawTracking(img, objects,  directions):
    
    colors = np.zeros([3,3])
    colors[2] = [0, 0, 255]
    colors[1] = [50, 50, 50]
    colors[0] = [255, 0, 0]


    for key, rect in objects.items():
        direction = directions[key]
        p1 = (int(rect[0]), int(rect[1]))
        p2 = (int(rect[2]), int(rect[3]))
        cv2.rectangle(img, p1, p2, colors[direction+1], 2)
        cv2.putText(img, str(key), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 2)

    return img

def draw_predictions(img, class_ids, vehicles, classes_description):
    for i, rect in enumerate(vehicles):
        img = draw_prediction(img, class_ids[i], rect[0], rect[1], rect[2], rect[3], classes_description)
    
    return img


        

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h, classes_description):

    label = str(classes_description[class_id])
    
    
    size = (x_plus_w - x) * (y_plus_h - y)

    color = (40, 40, min(size/100, 255))
    
    x = int(x)
    y = int(y)
    x_plus_w = int(x_plus_w)
    y_plus_h = int(y_plus_h)
    
    p1 = (x, y)
    p2 = (x_plus_w, y_plus_h)

    cv2.rectangle(img, p1, p2, color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def update_fps(tic, fps):
    toc = time.time()
    curr_fps = 1.0 / (toc-tic)
    fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
    tic = toc
    return tic, fps

def filter_not_detected(tracked, detected):
    deleteindexes = []
    for key, car in tracked.items():
        if car not in detected:
            deleteindexes.append(key)
            
    for key in deleteindexes:
        tracked.pop(key)        
    return tracked

def detect_and_track(frame, trt_yolo, classes, tracker, hw_func):
        frame_detection = frame.copy()
        rects, class_ids = hw_func.detect_cars(frame_detection, trt_yolo)
        frame_detection = draw_predictions(frame_detection, class_ids, rects, classes)
        
        
        poststamps=[]
        leaving = []
        entering = []
        
        for rect in rects:
              x = int(rect[0])
              y = int(rect[1])
              x_plus_w = int(rect[2])
              y_plus_h = int(rect[3])
              leaving.append(False)
              entering.append(False)
              
              if x<0 :
                  entering[-1] = True
                  x = 0
              if x> frame.shape[1]:  
                  leaving[-1] = True
              if y<0:
                  y = 0
              if y_plus_h>frame.shape[0]:
                  if x>int(frame.shape[1]/2):
                      leaving[-1] = True
                  else:
                      entering[-1] = True
                      
                  
              poststamps.append(frame[y:y_plus_h, x:x_plus_w])

        ids = tracker.update(rects, class_ids, poststamps, leaving, entering)
        ids = filter_not_detected(ids.copy(), rects)
        directions = tracker.returnDirections()
        
        if VIBRATE:
            vibrate.perform_output(ids, directions)
        
        frame_tracking = frame.copy()
        frame_tracking = drawTracking(frame_tracking, ids, directions)
        #storage.update(frame_nr, ids, directions)
        return frame_detection, frame_tracking
    



    
        
        
