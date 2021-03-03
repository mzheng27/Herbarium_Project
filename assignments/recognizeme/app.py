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

@app.route('/signup')
def signup():
	return render_template('/signup.html')

@app.route('/make_user', methods=['POST'])
def make_user():
	user = json.loads(request.form['user_info'])

	with open('users.json') as f:
		users = json.load(f)

	encoded_image = user['image'].split(",")[1]
	binary = BytesIO(base64.b64decode(encoded_image))
	image = Image.open(binary).convert("RGB")

	image_url = 'images/' + user['name'] + '.jpeg'
	image.save(image_url)

	users[user['name']] = {
		'name': user['name'],
		'image': image_url
	}

	with open('users.json', 'w') as f:
		json.dump(users, f)

	f.close()

	return render_template('./index.html')

@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
	def extract_face(filename, required_size=(224, 224)):
		pixels = pyplot.imread(filename)
		app.logger.info(str(pixels == None))
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

	app.logger.info("Initializing Program...")
	frame_uris = json.loads(request.form['video_feed'])
	app.logger.info(len(frame_uris))

	with open('users.json') as f:
		users = json.load(f)

	names = list(users.keys())

	user_images = np.array([extract_face(users[name]['image']) for name in names])

	# recorded_face = extract_face(user_images)
	model_scores_recorded = get_model_scores(user_images)

	pictures = []
	logged_in = False
	app.logger.info("Translating video...")
	for i in frame_uris:
		encoded_image = frame_uris[i].split(",")[1]
		binary = BytesIO(base64.b64decode(encoded_image))
		image = Image.open(binary)
		image = image.convert("RGB")
		pictures.append(np.array(image))
	pictures = np.array(pictures)
	frames = []
	i = 0
	pbar = tqdm.tqdm(total = len(pictures)//3)

	while i < len(pictures):
		frames.append(extract_face_video(pictures[i]))
		i += 3
		pbar.update(1)

	app.logger.info("Retrieving scores...")
	app.logger.info("Input shape is: " + str(np.array(frames).shape))

	your_face_score = get_model_scores(np.array(frames))

	result = []
	for i, score in enumerate(model_scores_recorded):
		result.append((names[i], min([cosine(i, score) for i in your_face_score])))

	app.logger.info(result)

	identified_user = min(result, key = lambda t: t[1])

	if identified_user[1] < 0.4:
		return render_template("logged_in.html", name=identified_user[0])
    # Pass in data to ML model and check if the user has been authorized or not
	else:
		return render_template("unauthorized.html")


if __name__ == "__main__":
    app.run(debug=True)
