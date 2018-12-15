from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory
import os
from os.path import join, dirname, realpath
import subprocess
import re
import os
import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
import pytesseract
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path, convert_from_bytes
import ntpath
import shutil

extensions = ('.pdf')
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import custom 



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_WEIGHTS_PATH = "./logs/mask_rcnn_table_0003.h5"  # TODO: update this path if needed
config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, "customImages")
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load validation dataset
dataset = custom.CustomDataset()
dataset.load_custom(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

# print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)


# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
# print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)


image_id = random.choice(dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]



def get_string(filename):
    # global image
    print(filename)
    image = skimage.io.imread(filename)
    results = model.detect([image], verbose=1)

        # Display results
    ax = get_ax(1)
    r = results[0]

    for bbox in r['rois']:

    # print(bbox)
        x1 = int(bbox[1])
        y1 = int(bbox[0])
        x2 = int(bbox[3])
        y2 = int(bbox[2])
        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 0), 0, 1)
        width = x2 - x1 
        height = y2 - y1 
        # print("x {} y {} h {} w {}".format(x1, y1, width, height))
        # crop_img = image[y1:y1+height, x1:x1+width]
        crop_img = image[y1:y2, x1:x2]
        # cv2.imshow("cropped", crop_img)
        cv2.imwrite('01.png',crop_img)
        string= pytesseract.image_to_string(Image.open("01.png"), config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/ -psm 4")
        print(string)
        print("END")
        return string
        # break
        

# the all-important app variable:
UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = set(['jpg','pdf'])

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/assets/<path:path>')
def send_js(path):
    return send_from_directory('assets', path)


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'file' in request.files:

        file = request.files['file']
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        ext = os.path.splitext(file_path)[-1].lower()
        all_strings=''
        if ext in extensions:
            images = convert_from_path(file_path, output_folder='./', fmt='png')
            for image in images:
                print("Generated"+image.filename)
                all_strings= all_strings + get_string(ntpath.basename(image.filename))
            
        else:
            shutil.copyfile(filename,'./')
            all_strings=get_string(filename)
        # Split strings
        list_strings = all_strings.splitlines()
        df = pd.DataFrame(list_strings)
                # df.replace('  ', np.nan, inplace=True)
                # df.dropna(inplace=True)
        print (df)
        writer = pd.ExcelWriter('output.xlsx')
        df.to_excel(writer,'Sheet1')
        writer.save()

        return render_template('index.html')

    return render_template('index.html')


@app.route('/table', methods=['GET', 'POST'])
def table():
    return render_template('index.html')


@app.route('/table-web', methods=['GET', 'POST'])
def table_web():
    return render_template('index.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=False, port=port)
