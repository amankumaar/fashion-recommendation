import os
from flask import Flask, jsonify, request
import tensorflow as tf
from PIL import Image
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import gdown

app = Flask(__name__)

# download the pickle files from Google Drive
url = 'https://drive.google.com/uc?id=1xBjCEPxlbNikF9bWjQup31L4aZiN5VO3'
output = 'image_features_embedding.pkl'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1pB_T6Ot5V74p2AvN0fIzPHx43GZJjqbT'
output = 'img_files.pkl'
gdown.download(url, output, quiet=False)

features_list = pickle.load(open("image_features_embedding.pkl", "rb"))
img_files_list = pickle.load(open("img_files.pkl", "rb"))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = Sequential([model, GlobalMaxPooling2D()])

def extract_img_features(uploaded_file, model):
    img = Image.open(uploaded_file.stream).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized


def recommendd(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices

def save_file(uploaded_file):
    try:
        uploaded_file.save(os.path.join("uploader", uploaded_file.filename))
        return 1
    except Exception as e:
        print(e)
        return 0


@app.route('/api/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('file')
    if uploaded_file:
        # extract features of uploaded image
        features = extract_img_features(uploaded_file, model)

        # recommend similar images
        img_indices = recommendd(features, features_list)

        img_files_list_fixed = [item.replace('\\', '/') for item in img_files_list]

        # return the list of similar images as response
        response = {
            'images': [img_files_list_fixed[img_indices[0][0]], img_files_list_fixed[img_indices[0][1]],
                       img_files_list_fixed[img_indices[0][2]], img_files_list_fixed[img_indices[0][3]],
                       img_files_list_fixed[img_indices[0][4]]]
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No file uploaded'})