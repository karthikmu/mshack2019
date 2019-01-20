import numpy as np
from flask import Flask, request, jsonify, current_app
import pickle
from sklearn.externals import joblib
import os
from keras.preprocessing import image
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import base64
import json
import tensorflow as tf


app = Flask(__name__)
with app.app_context():
    # within this block, current_app points to app.
    print (current_app.name)

model = joblib.load('cnn_model.pkl')

model.compile(loss="binary_crossentropy", optimizer='adam',metrics=["accuracy"])
graph = tf.get_default_graph()

# imgPath = os.getcwd() + '/49bb8caf-4c31-4bae-b345-1b03f5fd8e34.JPG'

@app.route('/predict',methods=['POST'])
def predict():
    global graph
    with graph.as_default():
        data = request.get_json()
        data = json.loads(data)
        imgPath = base64.b64decode(str(data['img']))
        filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgPath)

        inp = convert_image_to_array(filename)
        # print(type(inp), inp.shape)
        inp = inp[np.newaxis, :, :, :]
        # print (type(model))
        pd = model.predict(inp)
        label_name = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                      'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                      'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                      'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato_healthy']

        cls = np.where(pd[0, :] == 1)[0][0]

        print(label_name[cls])

        return jsonify(label_name[cls])

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, tuple((256, 256)))
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

if __name__ == "__main__":
    app.run(port=5000, debug=True)

