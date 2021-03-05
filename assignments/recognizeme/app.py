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
tf.compat.v1.disable_eager_execution()

app = Flask(__name__, template_folder="views")



class FastMTCNN(object):
	"""Fast MTCNN implementation."""
	def __init__(self, stride, resize=1, *args, **kwargs):
		self.stride = stride
		self.resize = resize
		self.mtcnn = mtcnn(*args, **kwargs)

	def __call__(self, frames):
		"""Detect faces in frames using strided MTCNN."""
		if self.resize != 1:
			frames = [cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize))) for f in frames]
		boxes, probs = self.mtcnn.detect(frames[::self.stride])
		faces = []
		pbar = tqdm.tqdm(total = len(frames))
		for i, frame in enumerate(frames):    
			box_ind = int(i / self.stride)
			if boxes[box_ind] is None:
				continue
			for box in boxes[box_ind]:
				box = [int(b) for b in box]
				faces.append(frame[box[1]:box[3], box[0]:box[2]])
			pbar.update(1)
		return faces
		
@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
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

		## If the face is tiny, you probably aren't trying to log on to your laptop
		full_img = Image.fromarray(pixels)
		print((width*height) / (full_img.size[0]*full_img.size[1]))
		if width*height < full_img.size[0]*full_img.size[1]*0.9:
			return [-1]

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

	app.logger.info("Initializing Program...")
	frame_uris = json.loads(request.form['video_feed'])
	app.logger.info(len(frame_uris))
	recorded_face = extract_face('jasonli.jpg')
	model_scores_recorded = get_model_scores(recorded_face)
	pictures = []
	logged_in = False
	app.logger.info("Translating video...")

	# Instantiate background subtractor and kernel
	back_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=50, detectShadows=False)    
	kernel = np.ones((30,30),np.uint8)


	for i in frame_uris:
		encoded_image = frame_uris[i].split(",")[1]
		binary = BytesIO(base64.b64decode(encoded_image))
		image = Image.open(binary)
		image = image.convert("RGB")
		# image.save("temp"+str(i)+".jpg")

		## Background subtraction on the images
		tmp = np.float32(image)
		fg_mask = back_sub.apply(tmp)
		fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
		fg_mask = cv2.medianBlur(fg_mask, 5) 
		## Threshold them
		_, fg_mask = cv2.threshold(fg_mask,127,255,cv2.THRESH_BINARY)
		img_x = fg_mask.shape[0]
		img_y = fg_mask.shape[1]

		## Check 1: See if bounding rect is too large/takes up much of the frame
		x,y,w,h = cv2.boundingRect(fg_mask)
		cpy = fg_mask.copy()
		cpy = cv2.rectangle(cpy, (x, y), (x+w, y+h), (255, 0, 0), 2)
		# savepath = "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/herbarium/Herbarium_Project/assignments/recognizeme/rect"+str(i)+".jpg"
		# cv2.imwrite(savepath, cpy)
		if (w*h) > (img_x*img_y*0.75):# or ((w*h) < (img_x*img_y*0.2)):
			continue
		## Check 2: If mask takes up much of bounding box
		temp = [1 if p > 0 else 0 for x in fg_mask for p in x]
		if w!=0 and h!=0:
			if (sum(temp)/(w*h)) > 0.65:
				continue 

		## Find contours
		contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contour_output = np.zeros(np.shape(fg_mask), dtype='uint8') # new blank canvas

		if len(contours) != 0:
			cv2.drawContours(contour_output, contours, -1, (255,0,0), 2)

		contour_area = sum([cv2.contourArea(contour) for contour in contours])
		pct_area = contour_area / (fg_mask.shape[0]*fg_mask.shape[1]) # Contour area's percent of frame

		## Check 3: contour area vs bounding rect
		if (contour_area > (w*h*0.7)): # or contour_area < (w*h*0.2)):
			continue

		# savepath = "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/herbarium/Herbarium_Project/assignments/recognizeme/temp"+str(i)+".jpg"
		# cv2.imwrite(savepath, fg_mask)

		pictures.append(np.array(image))
	pictures = np.array(pictures)
	frames = []
	i = 0
	pbar = tqdm.tqdm(total = len(pictures)//3)
	while i < len(pictures):
		tmp = extract_face_video(pictures[i])
		if len(tmp) > 1:
			frames.append(tmp)
		i += 3
		pbar.update(1)
	app.logger.info("Retrieving scores...")
	app.logger.info("Input shape is: " + str(np.array(frames).shape))

	# Check if any frames were saved. if not then unauthorized
	if (len(frames) < 7):
		return render_template("unauthorized.html")

	your_face_score = get_model_scores(np.array(frames))
	result = [cosine(i, model_scores_recorded) for i in your_face_score]
	app.logger.info(result)
	if min(result) < 0.4:
		return render_template("logged_in.html")
    # Pass in data to ML model and check if the user has been authorized or not
	else:
		return render_template("unauthorized.html")




if __name__ == "__main__":
    app.run(debug=True)
