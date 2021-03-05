# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>

# References 1: https://www.sitepoint.com/keras-face-detection-recognition/
# References 2: https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
# References 3: https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution
#MIT License
#
#Copyright (c) 2018 Ashutosh Pathak

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
import base64
from io import BytesIO
from PIL import Image
from binascii import a2b_base64
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import skimage.io as ski
from flask import Flask,request,jsonify, render_template
import numpy as np
import cv2
import base64
import json
import numpy as np
import keras_vggface
import mtcnn
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from scipy.spatial.distance import cosine
import tensorflow as tf
import tqdm
from facenet_pytorch import MTCNN as mtcnn
import torch


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml') 

def extract_face(filename, required_size=(224, 224)):
	pixels = pyplot.imread(filename)
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def extract_face_video(pixels, required_size = (224,224)):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# define our extractor
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

def get_model_scores(faces):
	if len(faces.shape) == 3:
		samples = asarray([faces], 'float32')
		samples = preprocess_input(samples, version=2)
	else:
		samples = asarray(faces, 'float32')
		samples = preprocess_input(samples, version=2)
    # perform prediction
	model = VGGFace(model='resnet50',include_top=False, input_shape=(224, 224, 3), pooling='avg')
	return model.predict(samples)

def extra_all_faces_from_video(pictures,frequency):
		frames = []
		i = 0
		pbar = tqdm.tqdm(total = len(pictures)//frequency)

		while i < len(pictures):
			frames.append(extract_face_video(pictures[i]))
			i += frequency
			pbar.update(1)
		return frames

def translate_video(frame_uris):
	pictures = []
	for i in frame_uris:
		encoded_image = frame_uris[i].split(",")[1]
		binary = BytesIO(base64.b64decode(encoded_image))
		image = Image.open(binary)
		image = image.convert("RGB")
		pictures.append(np.array(image))
	pictures = np.array(pictures)
	return pictures

def get_user_score():
	with open('users.json') as f:
		users = json.load(f)
	names = list(users.keys())
	user_images = np.array([extract_face(users[name]['image']) for name in names])
	model_scores_recorded = get_model_scores(user_images)
	return model_scores_recorded,names
	
def detect_eye_open(frame):
	#Initializing the face and eye cascade classifiers from xml files 
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.03,
            minNeighbors=5,
            minSize=(30, 30),
            # flags = cv2.CV_HAAR_SCALE_IMAGE
        )
	frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
	eyes = eye_cascade.detectMultiScale(
                frame,
                scaleFactor=1.03,
                minNeighbors=5,
                minSize=(30, 30),
                # flags = cv2.CV_HAAR_SCALE_IMAGE
            )
	if len(eyes) == 0:
		return 0
	else:
		return 1
