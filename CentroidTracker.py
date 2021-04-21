#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:46:53 2021

@author: marrit
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=3):
        self.nextObjectID=0
        self.objects = OrderedDict()
        self.sizes = OrderedDict()
        self.disappeared = OrderedDict()
        self.direction = OrderedDict()
        self.directionCount = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid, size):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.direction[self.nextObjectID] = 0
        self.directionCount[self.nextObjectID] = 0
        self.sizes[self.nextObjectID] = size
        print('object ' + str(self.nextObjectID) + ' registered')
        self.nextObjectID += 1
        
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        print('object ' + str(objectID) + ' disappeared')
        
    def updateOrientation(self, newCoordinates, newSize, objectID):
        if newSize > self.sizes[objectID]:
            self.directionCount[objectID] +=1
        elif newSize < self.sizes[objectID]:
            self.directionCount[objectID] -=1
            
        if self.directionCount[objectID] >= 3:
           self.directionCount[objectID] = 3
           self.direction[objectID] = 1
        elif self.directionCount[objectID] <= -3:
           self.directionCount[objectID] = -3
           self.direction[objectID] = -1
        elif self.directionCount[objectID] == 0:
            self.direction[objectID] = 0
            
       
        
    def update(self, rects):
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
            inputSizes[i] = abs(endX-startX) * abs(endY-startY)
            
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputSizes[i])
                
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectSizes = list(self.sizes.values())
            objectDirections = list(self.direction.values())

            
            D1 = dist.cdist(np.array(objectCentroids), inputCentroids)
            D2 = dist.cdist(np.array(objectSizes), inputSizes)

           
            D2[D2>0] = 0
            D2[D2<0] = 400
            D = D1 + D2

            print('D1', D1)
            print('D', D)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            
            for(row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                
                objectID = objectIDs[row]
                self.updateOrientation(inputCentroids[col], inputSizes[col], objectID)
                self.objects[objectID] = inputCentroids[col]
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
                    self.register(inputCentroids[col], inputSizes[col])
        #print(self.direction)
        #print(self.objects)
        return self.objects
