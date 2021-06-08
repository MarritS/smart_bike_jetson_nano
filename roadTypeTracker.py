# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:20:02 2021

@author: marri
"""
import shared_space_detector
from RoadType import RoadType

class roadTypeTracker():
    def __init__(self):
        self.cnt = 0
        self.separated_bikepath= False
        self.roadTypeCounter = 0
    
    
    def update(self, frame):
            roadType = shared_space_detector.testSingleImage(frame, False)
            print(roadType)
            if roadType is not RoadType.SEPARATED:
                self.roadTypeCounter = 0
                self.separated_bikepath = False
            elif roadType is RoadType.SEPARATED:
                self.roadTypeCounter += 1
                if self.roadTypeCounter >= 3:
                    self.separated_bikepath = True
            return self.separated_bikepath