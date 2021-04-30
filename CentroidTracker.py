# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:42:45 2021

@author: marri
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:15:07 2021

@author: marri
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:35:00 2021

@author: marri
"""

"""
Created on Tue Apr 20 15:46:53 2021
@author: marrit
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import math


import numpy as np
import cv2

SIZE_DELTA = 0.1 
DISTANCE_DELTA = 0.6
DEBUG = False

class CentroidTracker():
    def __init__(self, maxDisappeared=4):
        self.nextObjectID=0
        self.objects = OrderedDict()
        self.yPositions = OrderedDict()
        self.rectangles = OrderedDict()
        self.sizes = OrderedDict()
        self.disappeared = OrderedDict()
        self.direction = OrderedDict()
        self.prepareToDeregister = OrderedDict()
        self.directionCountForward = OrderedDict()
        self.directionCountBackward = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.classes = OrderedDict()
        self.postStamps = OrderedDict()
        
    def register(self, centroid, size, class_id, poststamp):
        self.objects[self.nextObjectID] = centroid
        self.yPositions[self.nextObjectID] = []
        
        yPos = centroid[1] + int(size/2)
        for x in range(3):
            self.yPositions[self.nextObjectID].append(yPos)
        self.disappeared[self.nextObjectID] = 0
        self.direction[self.nextObjectID] = 0
        self.directionCountForward[self.nextObjectID] = 0
        self.directionCountBackward[self.nextObjectID] = 0
        self.prepareToDeregister[self.nextObjectID] = False
        self.sizes[self.nextObjectID] = []
        for x in range(3):
            self.sizes[self.nextObjectID].append(size)
        self.postStamps[self.nextObjectID] = poststamp
        self.classes[self.nextObjectID] = class_id
        print('object ' + str(self.nextObjectID) + ' registered')
        self.nextObjectID += 1
        
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.yPositions[objectID]
        del self.disappeared[objectID]
        del self.direction[objectID]
        del self.directionCountForward[objectID]
        del self.directionCountBackward[objectID]
        del self.sizes[objectID]
        del self.classes[objectID]
        del self.postStamps[objectID]
        del self.prepareToDeregister[objectID]
        print('object ' + str(objectID) + ' disappeared')
        

        
    def updateOrientation(self, newCoordinates, newSize, objectID, poststamp):
        newSize = newSize
        oldSize = self.sizes[objectID][0]
        difSize = newSize - oldSize
        
        notComingTowards = False
        comingTowards = False
        delta = SIZE_DELTA * oldSize
        delta = max(6, delta)
        delta = 6
        
        if newSize > delta + oldSize:
            comingTowards = True
        elif newSize + delta < oldSize:
            notComingTowards = True
        else:
            print('nothing')
            
        newY = newCoordinates[1] + (int(newSize/2))
        oldY = self.yPositions[objectID][0]
        delta = DISTANCE_DELTA * oldSize
        delta = max(60, delta)
        delta = 8
            
        
# =============================================================================
#         
#         if newY>oldY+delta:
#               comingTowards = True
#         elif newY + delta < oldY:
#             notComingTowards = True
#         else:
#             print('nothing')
#             
# 
#         
# =============================================================================
        if notComingTowards is True:
            self.directionCountBackward[objectID] += 1
        else:
            self.directionCountBackward[objectID] -= 1
        
        if comingTowards is True:
            self.directionCountForward[objectID] += 1
        else:
            self.directionCountForward[objectID] -= 1
            
        
        
            
        MAXCOUNT = 4

        if self.directionCountForward[objectID] >= MAXCOUNT:
           self.directionCountForward[objectID] = MAXCOUNT
           self.direction[objectID] = 1
           #if self.direction[objectID] != 1:
               #self.directionCountBackward[objectID] = 0
               #self.direction[objectID] = 1
        elif self.directionCountForward[objectID] <= 0:
            self.directionCountForward[objectID] = 0
            if self.direction[objectID] == 1:
                self.direction[objectID] = 0
           
                
               
        if self.directionCountBackward[objectID] >= MAXCOUNT:
           self.directionCountBackward[objectID] = MAXCOUNT
           self.direction[objectID] = -1
           #if self.direction[objectID] != -1:
               #self.direction[objectID] = -1
               #self.directionCountForward[objectID] = 0
        elif self.directionCountBackward[objectID] <= 0:
            self.directionCountBackward[objectID] = 0
            if self.direction[objectID] == -1:
                self.direction[objectID] = 0

            
       
        
    def dominantColor(self, img):
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        dominant = palette[np.argmax(counts)]
        return dominant
    
    
        
    def update(self, rects, classes, poststamps, leaving, entering):
      
        for objectID in list(self.prepareToDeregister.keys()):
            if self.prepareToDeregister[objectID] is True:
                    self.deregister(objectID)
        
        
        
        
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                    
                
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 2), dtype = "int" )
        inputSizes = np.zeros((len(rects), 1), dtype = "int" )

        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputSizes[i] = abs(endY-startY)
            
        inputSizes = np.atleast_1d(np.squeeze(inputSizes))
        
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputSizes[i], classes[i], poststamps[i])
                
        else:
            objectIDs = list(self.objects.keys())
            objectClasses = list(self.classes.values())
            objectCentroids = list(self.objects.values())
            objectDirections = list(self.direction.values())
            objectSizes = list(self.sizes.values())
            objectPoststamps = list(self.postStamps.values())

            objectSizesLast = []
            for sizeList in objectSizes: (objectSizesLast.append(sizeList[-1]))
            objectSizesLast = np.array(objectSizesLast)
            

            

            D1 = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            D3 = np.zeros(D1.shape)
            for i in range(D3.shape[0]):
                for j in range(D3.shape[1]):
                    D3[i][j] = inputSizes[j] - objectSizesLast[i] 
            
            for z in range(D3.shape[0]):
                if objectDirections[z] == 0:
                    D3[z][:] = abs(D3[z][:])
                else:
                    D3[z][:] = D3[z][:] * objectDirections[z] * -1
                
            D3[D3<-50] = -50     
                
            
            #objectSizesArray = np.atleast_1d(np.squeeze(np.array(objectSizes)))
            #D2 = objectSizesArray[:, None] - inputSizes[None, :]
            #direction_array = np.atleast_2d(np.array(list(self.direction.values())))
            #for i in range(len(direction_array)):
                #D2[:, i] = D2[:, i] * direction_array[i]
                
            #D2 = D2 * -1

           
            #D2[D2>0] = 0
            #D2[D2<0] = 400
           # D = D1 + D2
            #print('D1', D1)
            #print('D', D)
            #D = D1
            
            D2 = np.zeros(D1.shape)
            for i in range(D2.shape[0]):
                for j in range(D2.shape[1]):
                     objectID = objectIDs[i]
                     imA = objectPoststamps[i]
                     imB = poststamps[j]
                    #imB = cv2.resize(imB, dsize=(imA.shape[1], imA.shape[0]), interpolation=cv2.INTER_CUBIC)
                
                     domColorA = self.dominantColor(imA)
                     domColorB = self.dominantColor(imB)
                     domColorDif = max(abs(domColorA - domColorB))
                     D2[i][j] = domColorDif
                     #print('The color difference is: ', domColorDif)
                    
                    
            
            
            D = D1 + 7*D2 + 1.5*D3
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            for(row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                #if classes[col] != objectClasses[row]:
                    #continue
                objectID = objectIDs[row]    
                
                imA = objectPoststamps[row]
                imB = poststamps[col]
                sizeImA = math.sqrt(imA.shape[0] * imA.shape[1])
                sizeImB = math.sqrt(imB.shape[0] * imB.shape[1])
                #if (D[row][col]> 200):
                   #continue
                if DEBUG:
                    print('sizes are: ' + str(sizeImA) + ', ' + str(sizeImB))
                    print('distance is: ', D[row][col])
                    print('\nLocation:  ' + str(D1[row][col]) + 
                          ' Color: ' + str(D2[row][col]) + 
                          ' Direction' + str(D3[row][col]))
                
                    cv2.destroyWindow("Old post")
                    cv2.destroyWindow("New post")
                    cv2.imshow('Old post', imA)
                    cv2.imshow('New post', imB)
                    cv2.waitKey(0)
                
                

               # 
                #s = ssim(imA, imB, multichannel=True)
                #print("Similarity score for tracked object " + str(objectID) + " is: " +str(s))
                
                #if s < 0.3:
                    
                     #continue
           
                #print("Sizes for id " + str(objectID) + " are " + str(self.sizes[objectID]))
                
                if leaving[col] is True:
                    self.prepareToDeregister[objectID] = True
                    print('DEREGISTER CAR BECAUSE OF PASSING, ID: ', objectID)
                
                if entering[col] is False:
                    self.updateOrientation(inputCentroids[col], inputSizes[col], objectID, poststamps[col])
                self.objects[objectID] = inputCentroids[col]
                self.postStamps[objectID] = poststamps[col]
                self.sizes[objectID].pop(0)
                self.sizes[objectID].append(inputSizes[col])
                yPos = inputCentroids[col][1] + int(inputSizes[col]/2)
                self.yPositions[objectID].pop(0)
                self.yPositions[objectID].append(yPos)
                
                self.disappeared[objectID] = 0
                
                
                
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputSizes[col], classes[col], poststamps[col])

        return self.objects
    
    def returnDirections(self):
        return self.direction
