from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from colorization import preprocess, post_process, ailia, load_img
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'static/images/'

WEIGHT_PATH = 'colorizer.onnx'
MODEL_PATH = 'colorizer.onnx.prototxt'
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Initialize the model
net = ailia.Net(MODEL_PATH, WEIGHT_PATH)
net.set_input_shape((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess the image
        img = load_img(filepath)
        img_lab_orig, img_lab_rs = preprocess(img)

        # Inference
        out = net.predict({'input.1': img_lab_rs})[0]

        # Post-process and save the output image
        out_img = post_process(out, img_lab_orig)
        savepath = os.path.join(app.config['RESULT_FOLDER'], filename)
        plt.imsave(savepath, out_img)

        return jsonify({
            'uploaded_image_url': url_for('display_image', filename=filename),
            'colorized_image_url': url_for('static', filename='images/' + filename)
        })

@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
