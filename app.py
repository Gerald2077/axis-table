from flask import Flask, flash, request, redirect, url_for, render_template, session, send_from_directory
import os
from os.path import join, dirname, realpath
import subprocess
import re
# the all-important app variable:
UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = set(['jpg'])

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
