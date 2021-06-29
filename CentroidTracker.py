
"""
A tracker for traffic which uses location/color/size+direction as distance metric
Created on Tue Apr 20 15:46:53 2021
@author: Marrit Schellekens
contact: marritschellekens@yahoo.com
"""

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

SIZE_DELTA = 6
MAX_COUNT_DIRECTION = 4
#This option will show the matches that the tracker makes in separate windows. 
#Press key to continue if this is on. 
DEBUG = False

class CentroidTracker():
    def __init__(self, maxDisappeared=4):
        self.nextObjectID=0
        self.centroids = OrderedDict()
        self.rects = OrderedDict()
        self.sizes = OrderedDict()
        self.disappeared = OrderedDict()
        self.direction = OrderedDict()
        self.prepareToDeregister = OrderedDict()
        self.directionCountForward = OrderedDict()
        self.directionCountBackward = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.classes = OrderedDict()
        self.postStamps = OrderedDict()
        
    def register(self, centroid, size, class_id, poststamp, rect):
        self.centroids[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.direction[self.nextObjectID] = 0
        self.directionCountForward[self.nextObjectID] = 0
        self.directionCountBackward[self.nextObjectID] = 0
        self.prepareToDeregister[self.nextObjectID] = False
        self.sizes[self.nextObjectID] = []
        
        #We want to compare the size with the size of four frames ago. 
        #Therefore we need to keep track of four sizes. 
        #To avoid initialization problems we simply add current size four times. 
        for x in range(3):
            self.sizes[self.nextObjectID].append(size)
        self.postStamps[self.nextObjectID] = poststamp
        self.classes[self.nextObjectID] = class_id
        print('object ' + str(self.nextObjectID) + ' registered')
        self.nextObjectID += 1
        
        
    def deregister(self, objectID):
        del self.centroids[objectID]
        del self.disappeared[objectID]
        del self.direction[objectID]
        del self.directionCountForward[objectID]
        del self.directionCountBackward[objectID]
        del self.sizes[objectID]
        del self.rects[objectID]
        del self.classes[objectID]
        del self.postStamps[objectID]
        del self.prepareToDeregister[objectID]
        print('object ' + str(objectID) + ' disappeared')
        
        
    def updateOrientation(self, newCoordinates, newSize, objectID):
        #We use index 0 to create queue like behavior. 
        oldSize = self.sizes[objectID][0]
        
        if newSize > SIZE_DELTA + oldSize:
            self.directionCountForward[objectID] += 1
        else:
            self.directionCountForward[objectID] -= 1
            
            if newSize + SIZE_DELTA < oldSize:
                self.directionCountBackward[objectID] += 1
            else:
                self.directionCountBackward[objectID] -= 1
            
   

        if self.directionCountForward[objectID] >= MAX_COUNT_DIRECTION:
           self.directionCountForward[objectID] = MAX_COUNT_DIRECTION
           self.direction[objectID] = 1
        elif self.directionCountForward[objectID] <= 0:
            self.directionCountForward[objectID] = 0
            if self.direction[objectID] == 1:
                self.direction[objectID] = 0
           
                
               
        if self.directionCountBackward[objectID] >= MAX_COUNT_DIRECTION:
           self.directionCountBackward[objectID] = MAX_COUNT_DIRECTION
           self.direction[objectID] = -1
        elif self.directionCountBackward[objectID] <= 0:
            self.directionCountBackward[objectID] = 0
            if self.direction[objectID] == -1:
                self.direction[objectID] = 0

            
       
    #Calculate dominant color using k-means clustering    
    def dominantColor(self, img):
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        dominant = palette[np.argmax(counts)]
        return dominant
    
    def averageColor(self, img):
        colors = np.zeros(3)
        for i in range(0, 3):
            res = np.sum(img[:,:,0], 1)
            res2 = sum(res)
            res3 = res2 / (img.shape[0]*img.shape[1])
            colors[i] = res3
        return colors
    
    
    def calculateDistanceMatrix(self, inputCentroids, inputSizes, poststamps):
        objectCentroids = list(self.centroids.values())
        objectDirections = list(self.direction.values())
        objectSizes = list(self.sizes.values())
        objectPoststamps = list(self.postStamps.values())

        objectSizesLast = []
        for sizeList in objectSizes: (objectSizesLast.append(sizeList[-1]))
        objectSizesLast = np.array(objectSizesLast)
        
        
        #Calculate distance between all tracked objects and detected objects
        D_location = dist.cdist(np.array(objectCentroids), inputCentroids)
        
        D_color = np.zeros(D_location.shape)
        D_size = np.zeros(D_location.shape)
        
        #Calculate difference between dominant colors
        for i in range(D_color.shape[0]):
            for j in range(D_color.shape[1]):
                 imA = objectPoststamps[i]
                 imB = poststamps[j]
            
                 domColorA = self.averageColor(imA)
                 domColorB = self.averageColor(imB)
                 domColorDif = max(abs(domColorA - domColorB))
                 D_color[i][j] = domColorDif
                
        #Calculate size difference
        for i in range(D_size.shape[0]):
            for j in range(D_size.shape[1]):
                D_size[i][j] = inputSizes[j] - objectSizesLast[i] 
        
        #Take direction into account 
        #(if tracked vehicle is coming towards the biker, the detected image should be bigger)
        #(if tracked vehicle is going away from the biker, the detected image should be smaller)
        #(if tracked vehicle direction is unknown, size difference is absolute)
        for z in range(D_size.shape[0]):
            if objectDirections[z] == 0:
                D_size[z][:] = abs(D_size[z][:])
            else:
                D_size[z][:] = D_size[z][:] * objectDirections[z] * -1
            
        #Maximum 'bonus' should be -50 to avoid ridiculous matches.     
        D_size[D_size<-50] = -50     
            
        #Weights determined by trial and error, can be changed if desired. 
        D = D_location + 7*D_color + 1.5*D_size
        
        return D, D_location, D_color, D_size
        
        
    def update(self, rects, classes, poststamps, leaving, entering):
      
        #Remove ids present in list prepareToDergister. 
        #This list is used to remove objects from which we know they have left the image. 
        for objectID in list(self.prepareToDeregister.keys()):
            if self.prepareToDeregister[objectID] is True:
                    self.deregister(objectID)
        
        
        
        #If nothing was detected. 
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                    
                
            return self.centroids
        
        
        #Transform lists into numpy arrays. Transform coordinates of box into centroids.  
        inputCentroids = np.zeros((len(rects), 2), dtype = "int" )
        inputSizes = np.zeros((len(rects), 1), dtype = "int" )

        
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputSizes[i] = abs(endY-startY)
            
        #Make sure the dimension is correct.     
        inputSizes = np.atleast_1d(np.squeeze(inputSizes))
        
        #If we aren't tracking aything: 
        if len(self.centroids) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputSizes[i], classes[i], poststamps[i], rects[i])
                
        else:
            D, D_location, D_color, D_size = self.calculateDistanceMatrix(inputCentroids, inputSizes, poststamps)
            #Sort matrix so lowest scores are in front. 
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()
            objectIDs = list(self.centroids.keys())
            objectPoststamps = list(self.postStamps.values())
            
            
            for(row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]    
                
                
                if DEBUG:
                    imA = objectPoststamps[row]
                    imB = poststamps[col]
                    print('\nLocation:  ' + str(D_location[row][col]) + 
                          ' Color: ' + str(D_color[row][col]) + 
                          ' Direction' + str(D_size[row][col]))
                
                    cv2.destroyWindow("Old post")
                    cv2.destroyWindow("New post")
                    cv2.imshow('Old post', imA)
                    cv2.imshow('New post', imB)
                    cv2.waitKey(0)
                
                #Remove vehicles that are leaving the frame in the next iteration
                if leaving[col] is True:
                    self.prepareToDeregister[objectID] = True
                    print('DEREGISTER CAR BECAUSE OF PASSING, ID: ', objectID)
                
                #update Variables
                
                #To avoid wrong direction updates, we do not update when the car is entering the frame from the front of bike. 
                if entering[col] is False:
                    self.updateOrientation(inputCentroids[col], inputSizes[col], objectID)
                self.centroids[objectID] = inputCentroids[col]
                self.rects[objectID] = rects[col]
                self.postStamps[objectID] = poststamps[col]
                self.sizes[objectID].pop(0)
                self.sizes[objectID].append(inputSizes[col])
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            #When we detect less objects than we have tracked, update disappeared counter and deregister if necessary. 
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            #When we detect more objects than we have tracked, register new objects           
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputSizes[col], classes[col], poststamps[col], rects[col])

        return self.rects
    
    def returnDirections(self):
        return self.direction