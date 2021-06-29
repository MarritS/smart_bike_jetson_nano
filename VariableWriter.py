#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:07:34 2021

@author: marrit
"""
import time

class VariableWriter():
     def __init__(self, FPS, writer):
         self.start = time.time()
         self.frame_nr = 0
         self.last_frame = None
         self.FPS = FPS
         self.writer = writer
         
     def write(self, frame):
         
         
        self.last_frame = frame
        if self.last_frame is None:
            self.last_frame = frame
            
        t = time.time() - self.start
        if t <= self.frame_nr/self.FPS:
            self.writer.write(self.last_frame)
            self.frame_nr+=1
        else:
            while t > self.frame_nr/self.FPS:
                
                self.writer.write(self.last_frame)
                self.frame_nr += 1
                t = time.time() - self.start
        
        self.last_frame = frame
            
                 