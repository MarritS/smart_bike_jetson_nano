from utils.camera import add_camera_args, Camera
import argparse
import os
import cv2
from utils.yolo_with_plugins import get_input_shape, TrtYOLO


import cv2
from utils.camera import Camera



import os
import time

from utils.display import open_window, set_display, show_fps

import CentroidTracker
import pycuda.autoinit #This is needed for intilazing CUDA driver
import Jetson_functions as hw_functions
import helper_functions as functions
from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import get_input_shape, TrtYOLO



CONF_TH = 0.3

def parse_args(VIDEO_FILE, CAMERA=False):
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
class jetson:

	def __init__(self):
		self.cam = None

	def get_frame(self):       	 	
		img = self.cam.read()

		if img is None:
            		success = False
		else:
			success = True
		return img, success

	def init_cam(self, VIDEO_FILE, CAMERA):
		args = parse_args(VIDEO_FILE, CAMERA)
		if args.category_num <= 0:
			raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
		if not os.path.isfile('yolo/%s.trt' % args.model):
			raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

		self.cam = Camera(args)
		if not self.cam.isOpened():
        		raise SystemExit('ERROR: failed to open camera!')
		print('Initalized camera')
		return cam	

	def detect_cars(self, frame, trt_yolo):
    		rects, confs, class_ids = trt_yolo.detect(frame, CONF_TH)
    		return rects, class_ids


    
