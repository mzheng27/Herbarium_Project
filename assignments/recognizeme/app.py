# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
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

app = Flask(__name__, template_folder="views")


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
    frame_uris = json.loads(request.form['video_feed'])

    pictures = []
    for i in frame_uris:
        encoded_image = frame_uris[i].split(",")[1]
        binary = BytesIO(base64.b64decode(encoded_image))
        image = Image.open(binary).convert("L")
        pictures.append(np.asarray(image))

    # Numpy array holding gray-scale video frames as numpy arrays
    data = np.array(pictures)

    # Pass in data to ML model and check if the user has been authorized or not
    logged_in = True


    if logged_in:
        return render_template("logged_in.html")
    else:
        return render_template("unauthorized.html")
if __name__ == "__main__":
    app.run(debug=True)
