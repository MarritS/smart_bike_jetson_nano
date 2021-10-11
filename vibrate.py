# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:42:29 2021

@author: marri
"""
import math
import Serial

class Vibration():
	def __init__(self):
		self.last_output = 0
	
	def perform_output(self, rects, directions):
		max_size = 0
    
    	#Find max size
		for key, rect in rects.items():
			if directions[key] == 1:
				d_x = rect[2] - rect[0]
				d_y = rect[3] - rect[1]
				size = math.sqrt(d_x * d_y)
				if size>max_size:
					max_size = size
					print(size)
    
		if (max_size > 250):
			output = 3
		elif (max_size > 120):
			output = 2
		elif (max_size > 0):
			output = 1
		else:
			output = 0
		
		if output != self.last_output:
			Serial.write_read(str(output))
			self.last_output = output
    
	def close():
    		Serial.write_read(str(0))
    		Serial.close()
    
    
