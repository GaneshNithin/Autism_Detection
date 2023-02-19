import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array 
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_madel():
    global loaded_model
    json_file = open('model_MobileNet_32.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_MobileNet_32.h5")
    print("Model loaded")

def preprocess_image(image,target_size):
    if image.mode!="RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)

    return image

print("Loading model")
get_madel()

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image,target_size=(224,224))

    predction = loaded_model.predict(processed_image).tolist()

    response = {
        'prediction':{
            'autistic':predction[0][0],
            'nonAutistic':predction[0][1]
        }
    }
    return jsonify(response)
