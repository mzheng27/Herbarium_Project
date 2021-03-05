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
from face_recognition_util import *
tf.compat.v1.disable_eager_execution()

app = Flask(__name__, template_folder="views")


		
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
	# change this to True if you are testing for extra credit
	extra_credit = False
	try:
		frequency = 1
		app.logger.info("Initializing Program...")
		frame_uris = json.loads(request.form['video_feed'])	
		model_scores_recorded, names = get_user_score()
		logged_in = False
		app.logger.info("Translating video...")
		pictures = translate_video(frame_uris)
		frames = extra_all_faces_from_video(pictures, frequency)
		app.logger.info("Retrieving scores...")
		app.logger.info("Input shape is: " + str(np.array(frames).shape))
		input_face_score = get_model_scores(np.array(frames))

		result = []
		for i, score in enumerate(model_scores_recorded):
			result.append((names[i], min([cosine(i, score) for i in input_face_score])))

		app.logger.info(result)
		identified_user = min(result, key = lambda t: t[1])

		if identified_user[1] < 0.4:
			if extra_credit == True:
				app.logger.info("Detecting eye blinks...")
				eye_detection_result = []
				while i < len(pictures):
					eye_detection_result.append(detect_eye_open(pictures[i]))
					i += frequency
				app.logger.info("Finish detecting eye blinks")
				app.logger.info(eye_detection_result)
				if max(eye_detection_result) - min(eye_detection_result) == 0:
					return render_template("unauthorized.html")
				else:
					return render_template("logged_in.html", name=identified_user[0])
			else:
				return render_template("logged_in.html", name=identified_user[0])
    	# Pass in data to ML model and check if the user has been authorized or not
		else:
			return render_template("unauthorized.html")
	except:
		return render_template("unauthorized.html")


if __name__ == "__main__":
    app.run(debug=True)
