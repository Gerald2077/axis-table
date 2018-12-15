import os
from pdf2image import convert_from_path, convert_from_bytes
rootdir = './uploads/'
extensions = ('.pdf')

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            print(file)
            if file != '.DS_Store':
                images = convert_from_path(os.path.join(
                    subdir, file), output_folder='./images-val', fmt='png')
                print(images[0].filename)
            else:
                print("gotcha")
