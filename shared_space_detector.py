# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:15:08 2021

@author: marri
"""

import pickle
import cv2
import numpy as np
import shared_space_detector
from scipy.ndimage import gaussian_filter
from RoadType import RoadType
DEBUG = False


def filter_img_bike(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red_A = np.array([170, 20, 0])
    upper_red_A = np.array([255, 255, 255])
    lower_red_B = np.array([0, 30, 0])
    upper_red_B = np.array([15, 255, 255])
    
    
    # preparing the mask to overlay
    maskA = cv2.inRange(hsv, lower_red_A, upper_red_A)
    maskB = cv2.inRange(hsv, lower_red_B, upper_red_B)
    mask = cv2.bitwise_or(maskA, maskB)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    if DEBUG:
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
         
        cv2.waitKey(0)
    return result

def filter_img_grass(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    lower_blue = np.array([30, 40, 0])
    upper_blue = np.array([80, 255, 255])
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    if DEBUG:
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
     
        cv2.waitKey(0)
    return result

def filter_img_road(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     

    lower_blue = np.array([15 , 0, 0])
    upper_blue = np.array([150, 65, 255])
    
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(frame, frame, mask = mask)
 
    if DEBUG:
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('result', result)
     
        cv2.waitKey(0)
    return result

def crop_img(frame):
    cropped_frame = frame[400:600, :]
    if DEBUG:
        cv2.imshow('cropped', cropped_frame)
        cv2.waitKey(0)
    return cropped_frame

def closing(frame):
    kernel = np.ones((10, 30),np.uint8)
    closed_frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    if DEBUG:
        cv2.imshow('closed', closed_frame)
        cv2.waitKey(0)
    return closed_frame
    
def opening(frame):
    kernel = np.ones((5, 5),np.uint8)
    opening_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    if DEBUG:
        cv2.imshow('opening', opening_frame)
        cv2.waitKey(0)
    return opening_frame

def return_sections(row):
    start_section = False
    sections = []
    
    for counter, cell in enumerate(row):
        if start_section is False and cell > 0:
            start_section = True
            start = counter
        elif start_section is True and cell == 0:
            start_section = False
            end = counter
            sections.append((start, end))
    
    if start_section is True:
        start_section = False
        end = counter
        sections.append((start, end))
        
    if len(sections) == 0:
        return None
            
    return sections

def returnLongestSection(sections):
    if sections == None:
        return None
    maxLength = 0
    for section in sections:
        length = section[1]-section[0]
        if length>maxLength:
            longestSection = section
            maxLength = length
            
    return longestSection

def comparison(roads, bikes, grasses, rown): 
    avg_bike_road_width = 1.8 * rown  + 10.9
    roadType = RoadType.SHARED
    
    road = returnLongestSection(roads)
    bike = returnLongestSection(bikes)
    grass = returnLongestSection(grasses)
    
    #if there are two pieces of road, the smaller of which is to the right and smaller than expected from main road.
    if roads != None:
        if len(roads) == 2 and bikes == None:
            for road_sec in roads:
                if road_sec is not road:
                    if road_sec[0] > road[0]:
                        size_road = road[1]-road[0]
                        size_road_sec = road_sec[1]-road_sec[0]
                        if size_road_sec<size_road and size_road<avg_bike_road_width:
                            return RoadType.SEPARATED
                        
    #Only bike path detected
    if road == None and bike != None:
        roadType = RoadType.SEPARATED
        return roadType
        
    #No roads detected    
    if road == None and bike == None:
        roadType = RoadType.UNKNOWN
        return roadType
          
    #If detected road is too small
    if road[1]-road[0] < avg_bike_road_width and bike == None:
        roadType = RoadType.SEPARATED
        return roadType
    
    #If detected road is big.
    if (bike == None and road[1]-road[0]>avg_bike_road_width):
        return RoadType.SHARED
    
    #If nothing is found
    if (bike == None or road == None or grass == None):
        return RoadType.UNKNOWN
   
    #If there is a stroke of grass between bike path and road. 
    for grass in grasses:
        for road in roads:
            if bike[0] < grass[0] and bike[1] < grass[1]:
                 if road[0]+12 > grass[0] and road[1] > grass[1]:
                     if road[1]-road[0]>10:
                         return RoadType.SEPARATED
                    
    

    return RoadType.UNKNOWN
            
        
def filter_area(frame):
    count = frame.any(axis=-1).sum()
    if count < 2000:
        frame = np.zeros(np.shape(frame))
    return frame

def blur_frame(frame):
    blurred = gaussian_filter(frame, sigma=(4, 4, 0))
    if DEBUG:
        cv2.imshow('blur', blurred)
        cv2.waitKey(0)
    return blurred
    
            
def draw_lines(frame_road, frame_bike, frame_grass):
    rown = 0
    cnt_separated = 0
    cnt_shared = 0
    cnt_unknown = 0
    
    
    frame_road = filter_area(frame_road)
    frame_bike = filter_area(frame_bike)
    frame_grass = filter_area(frame_grass)
    
    
    
    while rown < 200:
        row = frame_road[rown, :, 0]
        sections_road = return_sections(row) 
        
        row = frame_bike[rown, :, 0]
        sections_bike = return_sections(row) 
        
        row = frame_grass[rown, :, 0]
        sections_grass = return_sections(row) 
    
        
        
        roadType = comparison(sections_road, sections_bike, sections_grass, rown)
        if roadType == RoadType.SEPARATED:
            cnt_separated += 1
        elif roadType == RoadType.SHARED:
            cnt_shared +=1
        elif roadType == RoadType.UNKNOWN:
            cnt_unknown += 1
    
       
           
        
        rown += 1
      
    total = cnt_shared + cnt_unknown + cnt_separated
    prop_shared = cnt_shared/total
    prop_separated = cnt_separated/total
    


    if (cnt_unknown>cnt_shared and cnt_unknown > cnt_separated):
        if (prop_shared>0.33 and prop_separated < 0.15):
            roadType = RoadType.SHARED
        elif (prop_separated > 0.33 and prop_shared < 0.15):
            roadType = RoadType.SEPARATED
        else: 
            roadType = RoadType.UNKNOWN
    elif (cnt_separated>=cnt_shared):
        roadType = RoadType.SEPARATED
    elif (cnt_shared >= cnt_separated ):
        roadType = RoadType.SHARED
    
        
    if DEBUG:          
        print('Shared: ' + str(cnt_shared))
        print('Separated: ' + str(cnt_separated))
        print('Unknown: ' + str(cnt_unknown))    
        
        print('Road Type is:', str(roadType))
    return roadType
    

def testSingleImage(frame, DEBUG):
    #DEBUG = True
    
    
    test_frame = (cv2.resize(frame, (800, 600))).copy()
    test_frame = crop_img(test_frame)  
    test_frame = blur_frame(test_frame)
    
    
    test_frame_road = filter_img_road(test_frame)
    test_frame_road = opening(test_frame_road)
    test_frame_road = closing(test_frame_road)
    test_frame_road[np.where((test_frame_road>[0,0,0]).all(axis=2))] = [148,4,80]
    
    
    test_frame_grass = filter_img_grass(test_frame)
    test_frame_grass = opening(test_frame_grass)
    test_frame_grass = closing(test_frame_grass)
    test_frame_grass[np.where((test_frame_grass>[0,0,0]).all(axis=2))] = [79,231,80]
    
    
    
    test_frame_bike = filter_img_bike(test_frame)  
    test_frame_bike = opening(test_frame_bike)
    test_frame_bike = closing(test_frame_bike)
    test_frame_bike[np.where((test_frame_bike>[0,0,0]).all(axis=2))] = [239,157,80]
    
    
    roadType_pred = draw_lines(test_frame_road, test_frame_bike, test_frame_grass)

    if DEBUG:
      cv2.destroyAllWindows()
      
      cv2.imshow('img', cv2.resize(frame, (800, 600)))
      cv2.imshow('img_blur_crop', test_frame)
      cv2.imshow('road', test_frame_road)
      cv2.imshow('bike', test_frame_bike)
      cv2.imshow('grass', test_frame_grass)
      cv2.waitKey(0)
      
      cv2.destroyAllWindows()
    return roadType_pred
    
    
def findRoadType(frame_nr):
     frame = frames[frame_nr].copy()
     roadType_pred = testSingleImage(frame, False)

     return roadType_pred
 



def generateTestSet():
    frames = pickle.load( open( "frames.p", "rb" ) )
    clss = pickle.load( open( "clss.p", "rb" ) )
    
    #DEBUG = True
    #testSingleImage(180, True)
    #exit   
    
    DEBUG = False
    FP = []
    FN = []
    TN = []
    TP = []
    UNK_SHARED = []
    UNK_SEP = []
    
    
    for counter in range(0, len(frames)):
        print(str(counter) + ' of the ' + str(len(frames)) + ' processed.')
        #test_frame = frames[counter]
        roadType_pred = findRoadType(counter)
        
        if clss[counter] == 1:
            roadType_real = RoadType.SEPARATED
        else:
            roadType_real = RoadType.SHARED
        
        if roadType_pred == RoadType.UNKNOWN:
            if (roadType_real == RoadType.SHARED):
                UNK_SHARED.append(counter)
            else:
                UNK_SEP.append(counter)
        
        elif roadType_pred == RoadType.SEPARATED:
            if (roadType_pred == roadType_real):
                TP.append(counter)
            elif (roadType_pred != roadType_real):
                FP.append(counter)
        elif roadType_pred == RoadType.SHARED:
            if (roadType_pred == roadType_real):
                TN.append(counter)
            elif (roadType_pred != roadType_real):
                FN.append(counter)
        
            
       
        
        
        cv2.destroyAllWindows()
        
    print('FP: ', len(FP))
    print('FN: ', len(FN))
    print('TP: ', len(TP))
    print('TN: ', len(TN))

