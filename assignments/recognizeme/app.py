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

from flask import Flask,request,jsonify, render_template
import numpy as np
import cv2
import base64

app = Flask(__name__, template_folder="views")


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
    image_uri = request.form['video_feed']
    # https://gist.github.com/daino3/b671b2d171b3948692887e4c484caf47
    encoded_image = image_uri.split(",")[1]
    # binary = BytesIO(base64.b64decode(encoded_image))
    # image = Image.open(binary)
    # image.save("doink.png")
    binary_data = a2b_base64(encoded_image)

    fd = open('test.png', 'wb')
    fd.write(binary_data)
    fd.close()

    # image_array = np.asarray(image)

    # TODO: process the image as you see fit here to ensure the system recognizes
    # you and your teammates. Bonus points if you can prevent the system from being fooled by someone
    # holding up a photo of you or your teammates to the webcam, though this is not required.

    # For now, render the logged in page if the user is logged in.
    if image_uri:
        # im = Image.fromarray(image_array)
        # im.save("doink.png")
        return render_template("logged_in.html")
    return render_template("unauthorized.html")
if __name__ == "__main__":
    app.run(debug=True)
